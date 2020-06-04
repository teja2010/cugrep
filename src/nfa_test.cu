#include <stdio.h>
#include <stdint.h>
#include "common.h"

#ifdef NFA_TESTING

int main(int argc, char *argv[])
{
	if (argc != 4) {
		printf("Usage:\n");
		printf("./nfa_test [M|W] PATTERN STRING\n");
		exit(1);
	}

	if (strlen(argv[1]) != 1) {
		printf("[M|W]\n");
		exit(1);
	}

	bool matcharg = (argv[1][0] == 'M');

	uint8_t *nfa_blk = NULL;

	int len = build_nfa(argv[2], strlen(argv[2]), &nfa_blk);
	if (len < 0) {
		printf("build_nfa failed\n");
		return len;
	}

	if (match(nfa_blk, len, argv[3], strlen(argv[3])) == matcharg) {
		dprintf(2, "Pass\n");
		return 0;
	} else {
		dprintf(2, "Fail\n");
		return 0;
	}

}

#endif //NFA_TESTING
