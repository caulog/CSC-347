all: PiMC.c PiMC.cu
	g++ -o PiMC-cpu PiMC.c
	nvcc -o PiMC-gpu PiMC.cu
clean:
	rm -f PiMC-cpu
	rm -f PiMC-gpu