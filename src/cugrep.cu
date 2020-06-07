#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "common.h"

void print_help()
{
	printf("Usage: cugrep PATTERN FILE\n");
}

void check_config(struct config *c)
{
	//TODO: add checks on the pattern, etc.
}

struct config* read_config(int argc, char* argv[])
{
	if (argc != 2) {
		print_help();
		exit(1);
	}

	int pattern_len = strlen(argv[0]);

	struct config *c = (struct config *)calloc(1, sizeof(struct config));
	c->pattern = (char*)calloc(pattern_len, sizeof(char));
	memcpy(c->pattern, argv[0], pattern_len);
	printf("pattern %s\n", c->pattern);

	c->nfa_len = build_nfa(c->pattern, pattern_len, &c->nfa);
	if (c->nfa_len < 0) {
		printf("build_nfa failed\n");
		exit(1);
	} else
		printf("build_nfa done %x\n", c->nfa);

	c->filename = (char*)calloc(strlen(argv[1]), sizeof(char));
	memcpy(c->filename, argv[1], strlen(argv[1]));
	printf("filename %s\n", c->filename);

	c->fd = open(c->filename, O_RDONLY);
	if (c->fd == -1) {
		printf("cugrep: file open failed: %m\n");
		exit(1);
	}

	struct stat sb;
	if (fstat(c->fd, &sb) == -1) {
		printf("cugrep: file fstat failed: %m\n");
		exit(1);
	}
	c->file_size = sb.st_size;
	printf("Opened %s, size %d\n", c->filename, c->file_size);

	c->read_buf = (char*) mmap(NULL, c->file_size, PROT_READ|PROT_WRITE, MAP_PRIVATE,
				   c->fd, 0);
	if (c->read_buf == NULL) {
		printf("cugrep: file read failed: %m\n");
		exit(1);
	}
	c->file_offset = 0;

	c->offset = 0;
	check_config(c);
	return c;
}

#define OFFSET_SEARCH_LEN 64
__global__ void find_line_num_offsets(char *block, int block_size, int *num_offsets)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int blkidx = idx * OFFSET_SEARCH_LEN;
	if (blkidx > block_size)
		return;

	char *myBlock = block + blkidx;

	int ctr = 0;
	for(int i=0; i<OFFSET_SEARCH_LEN; i++) {
		if(myBlock[i] == '\n') {
			ctr++;
		}
	}
	num_offsets[idx] = ctr;
}

__global__ void fill_offsets(char *block, int block_size,
			     int *offset, int off_size,
			     int *global_ctr)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int blkidx = idx * OFFSET_SEARCH_LEN;
	if (blkidx == block_size)
		block[block_size-1] = '\0';

	if (blkidx >= block_size)
		return;

	char *myBlock = block + blkidx;
	int local_off[OFFSET_SEARCH_LEN];
	int ctr = 0;

	for(int i=0; i<OFFSET_SEARCH_LEN; i++) {
		if(myBlock[i] == '\n') {
			myBlock[i] = '\0';
			local_off[ctr] = blkidx+i+1;
#ifdef CUDA_TESTING
			printf("fill_offsets add: %d\n", blkidx+i+1);
#endif
			ctr++;
		}
	}

	if (ctr ==0)
		return;

	int offidx = atomicAdd(global_ctr, ctr);
	for(int i=0; i<ctr; i++) {
		offset[offidx+i] = local_off[i];
	}

}

__global__ void match_lines(char *block, int filesize, int size, int *offsets,
			    bool *found, uint8_t *nfa, int nfa_len)
{
	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_idx >= size) {
		return;
	}

	// strt matching
	char *str = block + offsets[thread_idx];
	int idx = 0;
	int nfa_idx = 0;
	int state = 0;
	bool reset = true;

	//if(str[0] == '#') // FIXME remove this
	//	return;

#ifdef CUDA_TESTING
	printf("thread_idx2: %d, offset %d, str:%s\n", thread_idx,
			offsets[thread_idx], str);
	printf("NFA_BLK:\n");
	for(uint i =0; i < nfa_len; i++) {
		printf(": %d %c %d\n",  NFA_CURR_STATE(nfa, i),
					NFA_MATCH_CHAR(nfa, i),
					NFA_NEXT_STATE(nfa, i));
	}
#endif

	while(str[idx] !='\0') {
		reset = true;
		for(int i=nfa_idx; i< nfa_len; i++) {
			if(NFA_CURR_STATE(nfa, i) != state)
				break;

#ifdef CUDA_TESTING
			printf("%d: %d =?= %d\n", i, state, NFA_CURR_STATE(nfa, i));
			printf("%d: %c =?= %c\n", i, str[idx], NFA_MATCH_CHAR(nfa, i));
#endif
			if(NFA_MATCH_CHAR(nfa, i) == str[idx] ) {
				state = NFA_NEXT_STATE(nfa, i);
#ifdef CUDA_TESTING
				printf("%c -> next %d\n", str[idx], state);
#endif
				reset = false;
				break;
			}
		}

		// nothing matched, reset state
		if (reset) {
#ifdef CUDA_TESTING
			printf("reset:%c\n", str[idx]);
#endif
			idx = idx - state;// reset idx
			state = 0;
			nfa_idx = 0;
		}

		if (state == 0xff) {
			found[thread_idx] = true;
#ifdef CUDA_TESTING
			printf("thread_idx: match %d\n", thread_idx);
#endif
			return;
		}

		while(nfa_idx < nfa_len && NFA_CURR_STATE(nfa, nfa_idx) < state) {
			nfa_idx++;
		}

		idx++;
	}

	found[thread_idx] = false;
#ifdef CUDA_TESTING
	printf("thread_idx: no match %d\n", thread_idx);
#endif
	return;
}

void cuErr(cudaError_t cerr, std::string err) {
	if (cerr != cudaSuccess) {
		fprintf(stderr, "%s Failed: %s", err.c_str(),
				cudaGetErrorString(cerr));
		exit(1);
	}
}

void find_line_simple(struct config *c, std::vector<int> &h_offsets)
{
	int len = 0;
	int prev = 0;
	for(int i=0; i<c->file_size; i++) {
		if (c->read_buf[i] == '\n') {
			h_offsets.push_back(prev);
			len++;
			c->read_buf[i] = '\0';
			prev = i+1;
			//printf("%d: %s\n", h_offsets[len-1],
			//			c->read_buf + h_offsets[len-1]);
		}
	}
	c->read_buf[c->file_size-1] = '\0';
	//printf("\n");
}

//void print_matches(struct config *c, bool *h_found,
//			thrust::host_vector<int> h_offsets, int h_found_size)
void print_matches(struct config *c, bool *h_found, int *h_offsets, int h_found_size)
{
	c->read_buf[c->file_size-1] = '\0';

	printf("Search Results:\n");
	for(int i=0; i<h_found_size; i++) {

		//printf("%s: %s\n", h_found[i] ? "T :" : "F :",
		//		&c->read_buf[h_offsets[i]]);

		if (h_found[i] == false)
			continue;

		if(i+1 < h_found_size)
			c->read_buf[h_offsets[i+1]-1] = '\0';

		puts(&c->read_buf[h_offsets[i]]);

	}
}


void find_line_offsets(char *d_read_buf, struct config *c,
		  thrust::device_vector<int> &d_offsets,
		  int *off_size_p)
{
	int num_off_size = (c->file_size -1)/OFFSET_SEARCH_LEN+1;
	thrust::device_vector<int> d_num_offsets(num_off_size);

	dim3 threads = dim3(THREADS_NUM, 1);
	dim3 blocks = dim3((num_off_size -1)/THREADS_NUM+1, 1);
	find_line_num_offsets<<<blocks, threads>>>(
			d_read_buf,
			c->file_size,
			thrust::raw_pointer_cast(&d_num_offsets[0])
			);
	
	int off_size = thrust::reduce(d_num_offsets.begin(),
					d_num_offsets.end(), 0,
					thrust::plus<int>());
	*off_size_p = off_size;
	d_offsets.resize(off_size);
	
	int *global_ctr;
	cuErr(cudaMalloc(&global_ctr, sizeof(int)),
			"Alloc global_ctr");

	// start another kernel to fill h_offsets
	fill_offsets<<<blocks, threads>>>(
			d_read_buf,
			c->file_size,
			thrust::raw_pointer_cast(&d_offsets[0]),
			off_size,
			global_ctr
			);

	int h_global_ctr;
	cuErr(cudaMemcpy(&h_global_ctr, global_ctr, sizeof(int),
					cudaMemcpyDeviceToHost),
			"Memcpy h_global_ctr");

	if (h_global_ctr != off_size) {
		printf("Error h_global_ctr %d != h_off_size %d\n",
				h_global_ctr, off_size);
		exit(1);
	} else {
		printf("Success h_global_ctr %d == h_off_size %d\n",
				h_global_ctr, off_size);
	}

	// sort h_offsets
	thrust::sort(d_offsets.begin(), d_offsets.end());
}

// the main loop where for each section of the file:
//	1. find line endings. populate offsets
//	2. start the kernel sharing the block and offsets.
//	3. print the results
void start_kernels(struct config *c)
{
	char *d_read_buf;
	cuErr(cudaMalloc(&d_read_buf, c->file_size*sizeof(char)),
			"Allocate Readbuf");
	cuErr(cudaMemcpy(d_read_buf, c->read_buf, c->file_size*sizeof(char),
					cudaMemcpyHostToDevice),
			"Memcpy Readbuf");


	int h_off_size = 0;
	thrust::device_vector<int> d_offsets;
	find_line_offsets(d_read_buf, c, d_offsets, &h_off_size);


	uint8_t *d_nfa_blk;
	cuErr(cudaMalloc(&d_nfa_blk, c->nfa_len*sizeof(uint32_t)),
			"Alloc d_nfa");
	cuErr(cudaMemcpy(d_nfa_blk, c->nfa, c->nfa_len*sizeof(uint32_t),
				cudaMemcpyHostToDevice),
			"Mempcy d_nfa");

	bool *d_found;
	cuErr(cudaMalloc(&d_found, h_off_size*sizeof(bool)),
			"Alloc d_found");

	printf("Start kernel\n");

	dim3 threads = dim3(THREADS_NUM, 1);
	dim3 blocks = dim3((h_off_size-1)/THREADS_NUM+1, 1);
	match_lines<<<blocks, threads>>>(d_read_buf,
					 c->file_size,
					 h_off_size,
					 thrust::raw_pointer_cast(&d_offsets[0]),
					 d_found,
					 d_nfa_blk,
					 c->nfa_len);
	// wait for all kernels to complete

	printf("Print found\n");
	//print the line if found is true

	bool *h_found;
	cuErr(cudaMallocHost(&h_found, h_off_size*sizeof(bool)),
			"Alloc h_found");
	cuErr(cudaMemcpy(h_found, d_found, h_off_size*sizeof(bool),
				cudaMemcpyDeviceToHost),
			"Memcpy h_found");

	int *h_offsets = (int*)calloc(h_off_size, sizeof(int));
	int *d_off_p = thrust::raw_pointer_cast(&d_offsets[0]);
	cuErr(cudaMemcpy(h_offsets, d_off_p, h_off_size*sizeof(int),
				cudaMemcpyDeviceToHost),
			"Memcpy h_offsets");

	print_matches(c, h_found, h_offsets, h_off_size);
}

int main(int argc, char *argv[])
{
	struct config *c = NULL;
	char *print_buf = (char*)calloc(PRINT_BUF_SIZE, sizeof(uint8_t));

	c = read_config(argc-1, argv+1);

	/* for fast IO set to UNBUFFERED
	 */
	if (setvbuf(stdout, print_buf, _IOFBF, PRINT_BUF_SIZE) < 0) {
		printf("setvbuf failed");
		exit(1);
	}

	start_kernels(c);
}


