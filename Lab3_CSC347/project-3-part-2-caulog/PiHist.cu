/*
** Olivia Caulfield
** Cho
** CSC 347
** 3/16/23
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define BLOCK_DIM 16

__global__ void histMethod(int numDigits, int piArray[], int *numCount);

int main(int argc, char **argv){
    printf("Distribution of the Digits of Pi: CUDA Code\n");
    
    // create count array on host
    int numCount[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // create count array on device
    int* dev_numCount;
    cudaMalloc((void**) &dev_numCount, 10*sizeof(int));
    // copy host array to device array
    cudaMemcpy(dev_numCount, numCount, 10*sizeof(int), cudaMemcpyHostToDevice);
 
    // error check for correct number of command line arguments
    if (argc != 3){
        printf("Invalid number of command line arguments.\n");
        exit(100);
    }
    // File pointer to digits of pi file for CPU memory
    FILE *piFile;
    piFile = fopen(argv[1], "r");
    if(piFile == NULL){
        printf("File '%s' not found\n", argv[1]);
        exit (101);
    }
    // error check for argument being a positive integer
    int numDigits = atoi(argv[2]); 
    if (numDigits <= 0){
        printf("'%s' is not a positive integer.\n", argv[2]);
        exit(102);
    }

    // make array in host memory to store numDigits of pi
    int* piArray = (int*) malloc(numDigits*sizeof(int));
    char c;
    for (int i = 0; i < numDigits; i++){
        c = fgetc(piFile);
        if(c == EOF){
            printf("End of file!\n");
            numDigits = i;
            break;
        }
        piArray[i] = ((int)c)-48;
    }

    // make array in device memory to store numDigits of pi
    int* dev_piArray;
    cudaMalloc((void**) &dev_piArray, numDigits*sizeof(int));
    // copy host array to device array
    cudaMemcpy(dev_piArray, piArray, numDigits*sizeof(int), cudaMemcpyHostToDevice);

    int N = numDigits;
    // define grid and block size
    dim3 dimBlock(BLOCK_DIM); 
    dim3 dimGrid((int)ceil(N+1/BLOCK_DIM));

    // allocate timers
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    // start timer
    cudaEventRecord(start,0);

    // call monte carlo method kernel
    histMethod<<<dimGrid,dimBlock>>>(numDigits, dev_piArray, dev_numCount);
    //histMethod<<<1,N>>>(numDigits, dev_piArray, dev_numCount);
    cudaDeviceSynchronize();

    // copy device count array to host array 
    cudaMemcpy(numCount, dev_numCount, 10*sizeof(int), cudaMemcpyDeviceToHost);

    // stop timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float diff;
    cudaEventElapsedTime(&diff, start, stop);
    // deallocate timers
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("time spent: %f\n", diff);
    
    for (int i = 0; i < 10; i++){
        printf("%d: %d\n", i, numCount[i]);
    }

    // write time to file
    FILE *fp;
    fp = fopen("piCUDA.csv", "w");
    for (int i = 0; i < 10; i++){
        fprintf(fp, "%f\n", (double)numCount[i]/numDigits*100);
    }
    fclose(fp);
    fclose(piFile);
    
    free(piArray);
    cudaFree(dev_piArray);
    cudaFree(dev_numCount);
    exit(0);
}

__global__ void histMethod(int numDigits, int piArray[], int *numCount){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < numDigits){
        int num = piArray[i];
        atomicAdd(&numCount[num], 1);
        //printf("%d\n", numCount[num]);
        //printf("%d: %d\n", numCount[i], num);
        //atomicAdd(&numCount[num], 1);
    }
}