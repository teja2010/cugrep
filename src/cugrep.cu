#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <helper_cuda.h>
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

	struct config *c = (struct config *)calloc(1, sizeof(struct config));
	c->pattern = (char*)calloc(strlen(argv[0]), sizeof(char));
	memcpy(c->pattern, argv[0], strlen(argv[0]));
	printf("pattern %s\n", c->pattern);

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

void build_metadata(struct config *c)
{
	c->nfa = (struct NFA*) calloc(1, sizeof(struct NFA));
	//TODO read the regex and build the NFA.
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

__global__ void match_lines(char *block, int size, int *offsets, bool *found)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}

	found[idx] = (block[offsets[idx]] == 's');

	//printf("[%d]: %s\n", idx, block+offsets[idx]);
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

	char *d_read_buf;
	cudaMalloc(&d_read_buf, c->file_size*sizeof(char));
	cudaMemcpy(d_read_buf, c->read_buf, c->file_size*sizeof(char), cudaMemcpyHostToDevice);

	thrust::device_vector<int> d_offsets(h_offsets.begin(), h_offsets.end());
	thrust::device_vector<bool> d_found(h_offsets.size());

	dim3 threads = dim3(THREADS_NUM, 1);
	dim3 blocks = dim3((h_offsets.size()-1)/THREADS_NUM+1, 1);
	match_lines<<<blocks, threads>>>(d_read_buf, d_offsets.size(),
			thrust::raw_pointer_cast(&d_offsets[0]),
			thrust::raw_pointer_cast(&d_found[0]));
	// wait for all kernels to complete

	//print the line if found is true
	thrust::host_vector<bool> h_found = d_found;

	//int count = 0;
	printf("Search Results:\n");
	for(int i=0; i<h_found.size(); i++) {
		if (h_found[i] == false)
			continue;

		printf("%s\n", &c->read_buf[h_offsets[i]]);
		//count++;
	}
	//printf("%d\n", count);
}

int main(int argc, char *argv[])
{
	struct config *c = NULL;

	c = read_config(argc-1, argv+1);

	build_metadata(c);

	start_kernels(c);
}


