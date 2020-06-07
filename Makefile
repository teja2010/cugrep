

all:
	#nvcc -O2 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	#nvcc -O2 -DNFA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	nvcc -g -DNFA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	#nvcc -g -DNFA_TESTING=1 -DCUDA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	#nvcc -O2 -DNFA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/nfa_test src/nfa.cu src/nfa_test.cu
	#nvcc -g -DNFA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/nfa_test src/nfa.cu src/nfa_test.cu


#comment
