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


/* NFA (non-deterministic finite automata) to represent a limited regex expression.
 * supports:
 *	"e*"  : matches 0 or more of the preceding expression e
 *	"e+"  : matches 0 or more of the preceding expression e
 *	"e|f" : matches expression e or expression f
 *
 *   Future work: Add support for:
 *	"."   : match any character TODO
 *	"(e)" : enclose a multi-character expression e
 *
 * returns nfa_blk_p 's length on success
 *         -1 on failure
 */
int build_nfa(char *regex, int regex_len, uint8_t **nfa_blk_p) {

	uint8_t *nfa_blk = NULL;

	nfa_blk = (uint8_t*)calloc(400, sizeof(uint8_t));
	*nfa_blk_p = nfa_blk;

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
		      regex[idx] == '?' || regex[idx] == '|'  )) {
		      //regex[idx] == '(' || regex[idx] == ')')) {
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
			cs = NFA_CURR_STATE(nfa_blk, nfa_idx-1);
			mc = NFA_MATCH_CHAR(nfa_blk, nfa_idx-1);
			ns = cs;
			NFA_SET(nfa_blk, nfa_idx-1, cs, mc, ns);
			state_counter--;
		}
		break;

		case '+': {
			if (idx == 0) {
				printf("Invalid regex, + at begining\n");
				return -1;
			}

			int cs, mc, ns;
			cs = state_counter;
			mc = NFA_MATCH_CHAR(nfa_blk, nfa_idx-1);
			ns = cs;
			NFA_SET(nfa_blk, nfa_idx, cs, mc, ns);
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

			NFA_SET(nfa_blk, nfa_idx, curr_state, regex[idx], state_counter);
			nfa_idx++;
		}
		}

		idx++;
	}
	expression_end.push_back(state_counter);

	std::vector<uint32_t> temp_nfa = {};
	for(int i=0; i < nfa_idx ; i++) {
		uint32_t bb = ((uint32_t)NFA_CURR_STATE(nfa_blk, i) << 16) +
		              ((uint32_t)NFA_NEXT_STATE(nfa_blk, i) << 8) +
		              ((uint32_t)NFA_MATCH_CHAR(nfa_blk, i)     );
		int next_state = NFA_NEXT_STATE(nfa_blk, i);

		for (int st : expression_end) {
			if (next_state == st) {
				bb |= 0xff00;
				break;
			}
		}

		//printf("bb : %lx\n", bb);
		temp_nfa.push_back(bb);
	}
	std::sort(temp_nfa.begin(), temp_nfa.end());

	for(uint i =0; i < temp_nfa.size(); i++) {
		uint32_t tn = temp_nfa[i];
		//printf("tn : %lx\n", tn);
		NFA_SET(nfa_blk, i, (tn >> 16) & 0xff ,
		                    (tn) & 0xff , (tn >> 8) & 0xff);
	}

	//printf("NFA_BLK:\n");
	//for(uint i =0; i < temp_nfa.size(); i++) {
	//	printf(": %d %c %d\n",  NFA_CURR_STATE(nfa_blk, i),
	//				NFA_MATCH_CHAR(nfa_blk, i),
	//				NFA_NEXT_STATE(nfa_blk, i));
	//}

	return nfa_idx;
}

// the function logic that will be used to match lines in the kernel.
bool match(uint8_t *nfa, int nfa_len, char* str, int slen)
{
	int thread_idx = 0;

	int idx = 0;
	int nfa_idx = 0;
	int state = 0;
	bool reset = true;


	while(str[idx] !='\0') {
		reset = true;
		for(int i=nfa_idx; i< nfa_len; i++) {
			if(NFA_CURR_STATE(nfa, i) != state)
				break;

			if(NFA_MATCH_CHAR(nfa, i) == str[idx] ) {
				state = NFA_NEXT_STATE(nfa, i);
				reset = false;
				break;
			}
		}

		// nothing matched, reset state
		if (reset) {
			idx = idx - state;
			state = 0;
			nfa_idx = 0;
		}

		if (state == 0xff) {
			return true;
		}

		while(nfa_idx < nfa_len && NFA_CURR_STATE(nfa, nfa_idx) < state) {
			nfa_idx++;
		}

		idx++;
	}

	return false;
}
