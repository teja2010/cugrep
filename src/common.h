#ifndef __CUGREP_COMMON_H
#define __CUGREP_COMMON_H

#include <stdio.h>

struct config {
	char *pattern, *filename;
	int fd;

	char *read_buf;
	long file_offset;
	long file_size;
	//struct NFA* nfa;
	uint32_t *nfa;
	int *file_metadata; // offsets to read each line.
	int offset;
};
#define READ_BUF_SIZE 2000

//struct NFA {
//	int temp;
//	//TODO
//};

#define THREADS_NUM 32
//struct vector {
//	void *arr;
//	int size, capacity, element_size;
//}

int build_nfa(char *regex, int regex_len, uint32_t **nfa_blk_p);
bool match();

#endif //__CUGREP_COMMON_H
