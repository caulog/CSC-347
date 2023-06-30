/*
** Olivia Caulfield
** Cho
** CSC 347
** 3/16/23
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#define BLOCK_DIM 16

__global__ void mcMethod(int iter, float *randPoints, int* inside);

// kernel to calculate random numbers
__global__ void generateRandomNumbers(curandState_t* states, float* numbers, int n);

int main(int argc, char **argv){
    printf("Monte Carlo Method: CUDA Code\n");

    //for (int reps = 0; reps < 100000000){
    // error check for correct number of command line arguments
    if (argc != 2){
        printf("Invalid number of command line arguments.\n");
        exit(100);
    }
    // error check for argument being a positive integer
    int iter = atoi(argv[1]); 
    if (iter <= 0){
        printf("'%s' is not a positive integer.\n", argv[1]);
        exit(101);
    }
    // store the total number of points being calculated
    int total = iter;
    
    // host count for points inside circle
    int inside[]= {0};
    // device count for points inside circle 
    int* dev_inside;
    cudaMalloc((void**) &dev_inside, sizeof(int));
    // copy host array to device arry
    cudaMemcpy(dev_inside, inside, sizeof(int), cudaMemcpyHostToDevice);
    
    // define size of array
    int N = iter*2;

    // define grid and block size
    dim3 dimBlock(BLOCK_DIM); 
    dim3 dimGrid((int)ceil(N+1/BLOCK_DIM));

    // host array for points
    float *randPoints = (float*) malloc(N *sizeof(float));
    // initialize as zeros
    for (int i = 0; i < N; i++){ randPoints[i] = 0; }
    // device array for points
    float *dev_randPoints;
    cudaMalloc((void**) &dev_randPoints, N *sizeof(float));
    // copy the host array of zeros to the device 
    cudaMemcpy(dev_randPoints, randPoints, N*sizeof(float), cudaMemcpyHostToDevice);

    // device memory for curandState_t objects
    curandState_t* states;
    cudaMalloc(&states, N * sizeof(curandState_t));

    // generate random numers
    generateRandomNumbers<<<dimGrid, dimBlock>>>(states, dev_randPoints, N);
    cudaDeviceSynchronize();
    // copy to host
    cudaMemcpy(randPoints, dev_randPoints, N*sizeof(float), cudaMemcpyDeviceToHost);

    /**error check
    for (int i = 0; i < iter; i++){
        printf("%d (%f, %f)\n", i, randPoints[i*2], randPoints[i*2+1]);
    }*/

    // allocate timers
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    // start timer
    cudaEventRecord(start,0);

    // call monte carlo method kernel
    mcMethod<<<dimGrid,dimBlock>>>(iter, dev_randPoints, dev_inside);
    cudaDeviceSynchronize();
    // copy device to host
    cudaMemcpy(inside, dev_inside, sizeof(int), cudaMemcpyDeviceToHost);

    // stop timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float diff;
    cudaEventElapsedTime(&diff, start, stop);
    // deallocate timers
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("time spent: %f\n", diff);
   
    // calculate pi
    int in = inside[0];
    float pi = float(4 * in )/total;
    printf("pi estimated: %f\n", pi);

    
    // write time to file
    FILE *fp;
    fp = fopen("piCUDA.csv", "w");
    fprintf(fp, "%f\n", diff);
    fclose(fp);

    free(randPoints);
    cudaFree(dev_randPoints);
    cudaFree(dev_inside);
    cudaFree(states);

    exit(0);
}

__global__ void mcMethod(int iter, float *randPoints, int* inside){
    // get every other index
    int i = (threadIdx.x + blockDim.x * blockIdx.x)*2;
    int size = iter*2;

    // if the index is in bounds, calculate distance and use atomicAdd
    if(i < size-1){
        float distance = randPoints[i]*randPoints[i] + randPoints[i+1]*randPoints[i+1];
        if (distance <= 1){
            atomicAdd(&inside[0], 1);
        }
    }
}

__global__ void generateRandomNumbers(curandState_t* states, float* numbers, int n) {
    int i = (threadIdx.x + blockDim.x * blockIdx.x)*2; 

    // Generate a random number and store it in the numbers array
    if(i < n-1){
        // initialize curand (seed, sequence, offset, state)
        curand_init(1234, i, 0, &states[i]);
        curand_init(1234, i+1, 0, &states[i+1]);

        numbers[i] = curand_uniform(&states[i]);
        numbers[i+1] = curand_uniform(&states[i+1]);
        //printf("%d (%f, %f)\n", i, numbers[i], numbers[i+1]);
    }
}
