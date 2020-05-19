

all:
	nvcc -O2 -I/usr/local/cuda-10.1/samples/common/inc -o cugrep cugrep.cu nfa.cu
	#nvcc -g -I/usr/local/cuda-10.1/samples/common/inc -o cugrep cugrep.cu nfa.cu


#comment
