/* 
**  Olivia Caulfield
**  Cho
**  CSC 347
**  2/28/23 
*/

#include <stdio.h> 
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#define BLOCK_DIM 16

__global__ void matrixMult(float* res, float* a, float* b, int N);
__global__ void generateRandomNumbers(curandState_t* states, float* numbers, int n);


int main(int argc, char **argv){
    printf("Matrix Multiplication: Naive CUDA\n");

    ////////////////// error checking /////////////////////////
    // error check for correct number of command line arguments
    if (argc != 2){
        printf("Invalid number of command line arguments.\n");
        exit(100);
    }
    // error check for argument being a positive integer
    int N = atoi(argv[1]); 
    if (N <= 0){
        printf("'%s' is not a positive integer.\n", argv[1]);
        exit(101);
    }
    /////////////////////////////////////////////////////////////


    ///////////// Do for various values of N ///////////////
    //float timeToCSV[100];
    //int nToCSV[100];
    //for (int count = 0; count < 100; count++){
        //int N = (count + 1) * 100;
        //nToCSV[count] = N;
        
        //////////// Allocate Memory //////////////
        int size = N * N * sizeof(float);
        // host memory 
        float *m1 = new float[N*N];
        float *m2 = new float[N*N];
        float *m3 = new float[N*N];
        // device memory
        float *dev_m1;
        float *dev_m2;
        float *dev_m3;
        cudaMalloc((void**) &dev_m1, size);
        cudaMalloc((void**) &dev_m2, size);
        cudaMalloc((void**) &dev_m3, size);
        // device memory for curandState_t objects
        curandState_t* states;
        cudaMalloc(&states, N * N * sizeof(curandState_t));
        ////////////////////////////////////////////
    

        // define grid and block size
        dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); 
        dim3 dimGrid((int)ceil((N+BLOCK_DIM-1)/BLOCK_DIM), (int)ceil((N+BLOCK_DIM-1)/BLOCK_DIM));


        ///////// Call kernel to generate random numbers //////////
        // matrix 1
        generateRandomNumbers<<<dimGrid, dimBlock>>>(states, dev_m1, N);
        cudaDeviceSynchronize();
        cudaMemcpy(m1, dev_m1, size, cudaMemcpyDeviceToHost);
        // matrix 2
        generateRandomNumbers<<<dimGrid, dimBlock>>>(states, dev_m2, N);
        cudaDeviceSynchronize();
        cudaMemcpy(m2, dev_m2, size, cudaMemcpyDeviceToHost);
        ////////////////////////////////////////////////////////////

        /* print matrix 1 *****check
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                printf("%f\t", m1[i*N+j]);
            }
            printf("\n");
        }
        printf("\n");


        // print matrix 2 *****check
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                printf("%f\t", m2[i*N+j]);
            }
            printf("\n");
        }
        printf("\n");*/

        ////////////// CUDA Timers ////////////////
        // allocate timers
        cudaEvent_t start;
        cudaEventCreate(&start);
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        // start timer
        cudaEventRecord(start,0);

        ///////// call the matrix multiply kernel //////
        matrixMult<<<dimGrid,dimBlock>>>(dev_m1, dev_m2, dev_m3, N);
        // transfer result to the host
        cudaMemcpy(m3, dev_m3, size, cudaMemcpyDeviceToHost);
        /////////////////////////////////////////////////

        // stop timer
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        float diff;
        cudaEventElapsedTime(&diff, start, stop);
        //timeToCSV[count]= diff; 
        // deallocate timers
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        /////////////////////////////////////////////////

        /*for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                printf("%f\t", m3[i*N+j]);
            }
            printf("\n");
        }*/

        //open a file for write
        FILE *fp;
        fp = fopen("product.dat", "w");

        // write product to a file
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                fprintf(fp, "%f\t", m3[i*N+j]);
            }
            fprintf(fp, "\n");
        }

        // output time 
        printf("time: %f s\n", diff/1000.0);
        
        // free the memory 
        fclose(fp);
        free(m1);
        free(m2);
        free(m3);
        cudaFree(dev_m1);
        cudaFree(dev_m2);
        cudaFree(dev_m3);
    //}

    /*FILE *fpt;
    fpt = fopen("NvsTime.csv", "w+");
    for (int i = 0; i < 100; i++){
        fprintf(fpt, "%d, %f\n", nToCSV[i], timeToCSV[i]);
    }
    fclose(fpt);*/

    return 0;
}

__global__ void matrixMult(float* m1, float* m2, float* m3, int N) {
    // function that runs on GPU to do the multiplication
    int k;
    float sum = 0;

    int col = threadIdx.x + blockDim.x * blockIdx.x; 
    int row = threadIdx.y + blockDim.y * blockIdx.y; 
    
    if(col < N && row < N) { 
        for (k = 0; k < N; k++) {
            sum += m1[row * N + k] * m2[k * N + col]; 
        }
        m3[row * N + col] = sum; 
    }
}

__global__ void generateRandomNumbers(curandState_t* states, float* numbers, int N) {
    // Initialize the curandState_t object for each thread
    int col = threadIdx.x + blockDim.x * blockIdx.x; 
    int row = threadIdx.y + blockDim.y * blockIdx.y; 
    int idx = row * N + col;

    // initialize curand (seed, sequence, offset, state)
    curand_init(1234, idx, 0, &states[idx]);
    
    // Generate a random number and store it in the numbers array
    if (idx < N*N){
        numbers[idx] = curand_uniform(&states[idx]);  
    }
}