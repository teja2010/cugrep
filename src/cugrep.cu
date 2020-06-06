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

//#define OFFSET_SEARCH_LEN 10 /* length of blocks where we search for newline */
//__global__ void find_line_offsets(char *block, int block_size, int *offsets, int *offsets_num)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx > block_size)
//		return;
//
//	char *myBlock = block + idx;
//	int *myOffsets = offsets + idx;
//	int *myOffsets_num = offsets_num + idx;
//
//	int ctr = 0;
//	for(int i=0; i<OFFSET_SEARCH_LEN; i++) {
//		if(myBlock[i] == '\n') {
//			myOffsets[ctr] = idx+i;
//			ctr++;
//		}
//	}
//	*myOffsets_num = ctr;
//}
// some test code to match:
	//// pattern = ^s
	//found[idx] = (block[offsets[idx]] == 's');
	
	//// pattern = s
	//int start = offsets[idx];
	//bool  ff = false;
	//while(!ff && start < filesize && block[start] != '\0') {
	//	ff = ff || (block[start] == 's');
	//	start++;
	//}
	//found[idx] = ff;

	//printf("[%d]: %s\n", idx, block+offsets[idx]);

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
	printf("thread_idx2: %d, str %s\n", thread_idx, str);
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

// the main loop where for each section of the file:
//	1. find line endings. populate offsets
//	2. start the kernel sharing the block and offsets.
//	3. print the results
void start_kernels(struct config *c)
{
	std::vector<int> h_offsets;
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
	cudaError_t err;

	char *d_read_buf;
	err = cudaMalloc(&d_read_buf, c->file_size*sizeof(char));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate readbuf: %s\n",
				cudaGetErrorString(err));
		exit(1);
	}
	err = cudaMemcpy(d_read_buf, c->read_buf, c->file_size*sizeof(char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to memcpy readbuf: %s\n",
				cudaGetErrorString(err));
		exit(1);
	}

	uint8_t *d_nfa_blk;
	err = cudaMalloc(&d_nfa_blk, c->nfa_len*sizeof(uint32_t));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to alloc d_nfa: %s\n",
				cudaGetErrorString(err));
		exit(1);
	}
	err = cudaMemcpy(d_nfa_blk, c->nfa, c->nfa_len*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to memcpy d_nfa: %s\n",
				cudaGetErrorString(err));
		exit(1);
	}

	thrust::device_vector<int> d_offsets(h_offsets.begin(), h_offsets.end());
	thrust::device_vector<bool> d_found(h_offsets.size());

	printf("Start kernel\n");

	dim3 threads = dim3(THREADS_NUM, 1);
	dim3 blocks = dim3((h_offsets.size()-1)/THREADS_NUM+1, 1);
	match_lines<<<blocks, threads>>>(d_read_buf,
					 c->file_size,
					 d_offsets.size(),
					 thrust::raw_pointer_cast(&d_offsets[0]),
					 thrust::raw_pointer_cast(&d_found[0]),
					 d_nfa_blk,
					 c->nfa_len);
	// wait for all kernels to complete

	printf("Print found\n");
	//print the line if found is true
	thrust::host_vector<bool> h_found = d_found;

//	print_matches(h_found.data(),
//		      h_offsets.data(),
//		      h_found.size(),
//		      c->read_buf);


	//int count = 0;
	printf("Search Results:\n");
	//flockfile(stdout);
	for(int i=0; i<h_found.size(); i++) {
		if (h_found[i] == false)
			continue;

		//printf("%s\n", &c->read_buf[h_offsets[i]]);
		puts(&c->read_buf[h_offsets[i]]);

		//fputc('\n', stdout);
		//fputs_unlocked(&c->read_buf[h_offsets[i]], stdout);
		//fputc_unlocked('\n', stdout);
		//count++;
	}
	//funlockfile(stdout);
	//printf("%d\n", count);
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


