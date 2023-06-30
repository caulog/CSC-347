all: PiHist.c PiHist.cu
	g++ -o PiHist-cpu PiHist.c
	nvcc -o PiHist-gpu PiHist.cu
clean:
	rm -f PiHist-cpu
	rm -f PiHist-gpu