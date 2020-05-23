/* file that implements the comparing */
#include <stdio.h>
#include <stdint.h>
#include "common.h"
#include <vector>
#include <algorithm>
#include <stack>
#include <helper_cuda.h>

//struct nfa_state {
//	union {
//		uint32_t ignore_this;
//		struct {
//			uint8_t curr_state;
//			uint8_t match_char;
//			uint8_t next_state;
//			uint8_t again_ignore;
//		};
//	};
//};
#define NFA_CURR_STATE(b32) ((b32 >> 16) & 0xff)
#define NFA_MATCH_CHAR(b32) ((b32 >> 8) & 0xff)
#define NFA_NEXT_STATE(b32) ((b32) & 0xff)

#define NFA_SET(b32, cs, mc, ns)	\
	do {									\
		b32 = ( (((uint32_t)cs) << 16) | (((uint32_t)mc) << 8) |	\
				((uint32_t)ns) );				\
		if (true)							\
			printf("SET %d, %c, %d -> %x\n", cs, mc, ns, b32);	\
	}while(0);

/* NFA (non-deterministic finite automata) to represent a limited regex expression.
 * supports:
 *	"e*"  : matches 0 or more of the preceding expression e
 *	"e+"  : matches 1 or more of the preceding expression e
 *	"e|f" : matches expression e or expression f
 *	"(e)" : enclose a multi-character expression e
 *
 * returns nfa_blk_p 's length on success
 *         -1 on failure
 */
int build_nfa(char *regex, int regex_len, uint32_t **nfa_blk_p) {

	uint32_t *nfa_blk = NULL;
	//checkCudaErrors(cudaMallocHost(&nfa_blk, 100*sizeof(uint32_t)));
	nfa_blk = (uint32_t*)calloc(100, sizeof(uint32_t));

	int idx = 0;

	int nfa_idx = 0;
	int state_counter = 0;
	std::stack<int> expression_start = {};
	std::vector<int> expression_end = {};

	while(idx < regex_len) {

		// check validity of character
		// i.e. (a,z) || (A,Z) || {*, +, ?, (, ) }
		if (!((regex[idx] >= 'a' && regex[idx] <= 'z') ||
		      (regex[idx] >= 'A' && regex[idx] <= 'Z') ||
		      regex[idx] == '*' || regex[idx] == '+' ||
		      regex[idx] == '?' || regex[idx] == '|' ||
		      regex[idx] == '(' || regex[idx] == ')')) {
			printf("Invalid character %c(%d)\n",
					regex[idx], regex[idx]);
			return -1;
		}

		switch(regex[idx]) {
		case '*': {
			if (idx == 0) {
				printf("Invalid regex, * at begining\n");
				return -1;
			}

			int cs, mc, ns;
			cs = NFA_CURR_STATE(nfa_blk[nfa_idx-1]);
			mc = NFA_MATCH_CHAR(nfa_blk[nfa_idx-1]);
			ns = cs;
			NFA_SET(nfa_blk[nfa_idx-1], cs, mc, ns);
			state_counter--;
		}
		break;

		case '+': {
			if (idx == 0) {
				printf("Invalid regex, + at begining\n");
				return -1;
			}

			int cs, mc, ns;
			cs = state_counter++;
			mc = NFA_MATCH_CHAR(nfa_blk[nfa_idx-1]);
			ns = cs;
			NFA_SET(nfa_blk[nfa_idx], cs, mc, ns);
			nfa_idx++;
		}
		break;

		case '|': {
			if (idx == 0) {
				printf("Invalid regex, | at begining\n");
				return -1;
			}
			expression_end.push_back(state_counter);
		}
		break;

		default: {
			int curr_state = state_counter++;
			if (expression_start.size() == 0) {
				expression_start.push(curr_state);
			}

			if (idx > 0 && regex[idx-1] == '|') {
				curr_state = expression_start.top();
			}

			NFA_SET(nfa_blk[nfa_idx], curr_state, regex[idx], state_counter);
			nfa_idx++;
		}
		}

		idx++;
	}
	expression_end.push_back(state_counter);

	std::vector<uint32_t> temp_nfa = {};
	for(int i=0; i < nfa_idx ; i++) {
		uint32_t bb = nfa_blk[i];
		int next_state = bb&0xff;

		for (int st : expression_end) {
			if (next_state == st) {
				bb |= 0xff;
				break;
			}
		}

		temp_nfa.push_back(bb);
	}
	std::sort(temp_nfa.begin(), temp_nfa.end());

	for(int i=0; i < temp_nfa.size(); i++) {
		nfa_blk[i] = temp_nfa[i];

		uint32_t bb = nfa_blk[i];
		printf(": %d %c %d\n",  NFA_CURR_STATE(bb),
					NFA_MATCH_CHAR(bb),
					NFA_NEXT_STATE(bb));
	}

	return nfa_idx;
}

bool match(uint32_t *nfa, int nfa_len, char* str, int slen)
{

	return false;
}
