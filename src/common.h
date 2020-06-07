#ifndef __CUGREP_COMMON_H
#define __CUGREP_COMMON_H

#include <stdio.h>

struct config {
	char *pattern, *filename;
	int pattern_len;
	int fd;

	char *read_buf;
	long file_offset;
	long file_size;
	
	//NFA
	uint8_t *nfa;
	int nfa_len;

};
#define READ_BUF_SIZE 2000

//struct NFA {
//	int temp;
//	//TODO
//};
#define PRINT_BUF_SIZE 2046

#define THREADS_NUM 256
//struct vector {
//	void *arr;
//	int size, capacity, element_size;
//}

#define NFA_CURR_STATE(b32, idx) (b32[4*(idx)])
#define NFA_MATCH_CHAR(b32, idx) (b32[4*(idx) +2])
#define NFA_NEXT_STATE(b32, idx) (b32[4*(idx) +1])

#ifdef NFA_TESTING
#define NFA_SET(b32, idx, cs, mc, ns)					\
	do {								\
		b32[4*(idx)]   = cs & 0xff;				\
		b32[4*(idx)+2] = mc & 0xff;				\
		b32[4*(idx)+1] = ns & 0xff;				\
		printf("SET @%d: %d, %c -> %x\n", idx, cs, mc, ns);	\
	}while(0);

#else //NFA_TESTING
#define NFA_SET(b32, idx, cs, mc, ns)					\
	do {								\
		b32[4*idx] = cs & 0xff; 				\
		b32[4*idx+2] = mc & 0xff;				\
		b32[4*idx+1] = ns & 0xff;				\
	} while(0);
#endif //NFA_TESTING

int build_nfa(char *regex, int regex_len, uint8_t **nfa_blk_p);
bool match(uint8_t *nfa, int nfa_len, char* str, int slen);

void print_matches(bool *h_found, int *h_offsets, int size, char *read_buf);

#endif //__CUGREP_COMMON_H
