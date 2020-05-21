

all:
	nvcc -O2 -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu
	#nvcc -g -I/usr/local/cuda-10.1/samples/common/inc -o testing/cugrep src/cugrep.cu src/nfa.cu


#comment
