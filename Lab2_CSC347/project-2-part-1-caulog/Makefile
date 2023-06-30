all: MatrixMult.c MatrixMultNaive.cu MatrixMultTiled.cu
	g++ -o MatrixMult MatrixMult.c
	nvcc -o MatrixMultNaive MatrixMultNaive.cu
	nvcc -o MatrixMultTiled MatrixMultTiled.cu
clean:
	rm -f MatrixMult
	rm -f MatrixMultNaive
	rm - MatrixMultTiled