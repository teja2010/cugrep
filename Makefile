

all:
	#nvcc -O2 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	#nvcc -O2 -DNFA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	#g++ -O2 -fPIC -shared -nostartfiles -o print_lines.so src/print_lines.c
	#g++ -O2 -fPIC -nostartfiles -c  -o print_lines.o src/print_lines.c
	#nvcc -g -O2 -DNFA_TESTING=1 -L. -l:print_lines.so -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	#nvcc -g -O2 -DNFA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu print_lines.o
	nvcc -g -DNFA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	#nvcc -g -DNFA_TESTING=1 -DCUDA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	#nvcc -O2 -DNFA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/nfa_test src/nfa.cu src/nfa_test.cu
	nvcc -g -DNFA_TESTING=1 -I/usr/local/cuda-10.1/samples/common/inc -o testing/nfa_test src/nfa.cu src/nfa_test.cu


#comment
