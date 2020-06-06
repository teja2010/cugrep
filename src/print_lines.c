#include <stdio.h>

void print_matches(bool *h_found, int *h_offsets, int size, char *read_buf)
{
	printf("Search Results:\n");
	flockfile(stdout);
	for(int i=0; i<size; i++) {
		if (h_found[i] == false)
			continue;

		//printf("%s\n", &c->read_buf[h_offsets[i]]);
		//puts(&c->read_buf[h_offsets[i]]);
		//fputs(&c->read_buf[h_offsets[i]], stdout);
		//fputc('\n', stdout);
		fputs_unlocked(&read_buf[h_offsets[i]], stdout);
		fputc_unlocked('\n', stdout);
	}
	funlockfile(stdout);
}
