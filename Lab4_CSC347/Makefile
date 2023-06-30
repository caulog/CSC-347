all: nbody.cpp nbody.cu
	nvcc -o nbody-cpu nbody.cpp
	nvcc -o nbody-gpu nbody.cu
clean:
	rm -f nbody-cpu
	rm -f nbody-gpu
