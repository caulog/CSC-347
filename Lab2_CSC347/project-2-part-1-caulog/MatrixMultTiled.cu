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
#define TILE_WIDTH BLOCK_DIM

__global__ void matrixMult(float* m1, float* m2, float* m3, int N);
__global__ void generateRandomNumbers(curandState_t* states, float* numbers, int N);

int main(int argc, char **argv){
    printf("Matrix Multiplication: Tiled CUDA\n");

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

    //float timeToCSV[100];
    //int nToCSV[100];

    //for (int count = 0; count < 100; count++){
        //int N = (count + 1) * 100;
        //nToCSV[count] = N;
        
        int size = N * N * sizeof(float);
        
        // allocate host memory 
        float *m1 = new float[N*N];
        float *m2 = new float[N*N];
        float *m3 = new float[N*N];

        // allocate GPU memory
        float *dev_m1;
        float *dev_m2;
        float *dev_m3;
        cudaMalloc((void**) &dev_m1, size);
        cudaMalloc((void**) &dev_m2, size);
        cudaMalloc((void**) &dev_m3, size);

        curandState_t* states;
        // Allocate memory on the GPU for the curandState_t objects
        cudaMalloc(&states, N * N * sizeof(curandState_t));
    
        // define grid and block size
        dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); 
        dim3 dimGrid((int)ceil((N+BLOCK_DIM-1)/BLOCK_DIM), (int)ceil((N+BLOCK_DIM-1)/BLOCK_DIM));

        // Call the kernel function to generate the random numbers
        // matrix 1
        generateRandomNumbers<<<dimGrid, dimBlock>>>(states, dev_m1, N);
        cudaDeviceSynchronize();
        // copy back to host matrix 1
        cudaMemcpy(m1, dev_m1, size, cudaMemcpyDeviceToHost);
        // matrix 2
        generateRandomNumbers<<<dimGrid, dimBlock>>>(states, dev_m2, N);
        cudaDeviceSynchronize();
        // copy back to host matrix 2
        cudaMemcpy(m2, dev_m2, size, cudaMemcpyDeviceToHost);

        /** print check
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                printf("%f\t", m1[i*N+j]);
            }
            printf("\n");
        }
        printf("\n");


        // print check
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                printf("%f\t", m2[i*N+j]);
            }
            printf("\n");
        }
        printf("\n");*/

        // allocate timers
        cudaEvent_t start;
        cudaEventCreate(&start);
        cudaEvent_t stop;
        cudaEventCreate(&stop);

        // start timer
        cudaEventRecord(start,0);

        // call the matrix multiply kernel
        matrixMult<<<dimGrid,dimBlock>>>(dev_m1, dev_m2, dev_m3, N);

        // transfer res to the host
        cudaMemcpy(m3, dev_m3, size, cudaMemcpyDeviceToHost);

        // stop timer and print time
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        float diff;
        cudaEventElapsedTime(&diff, start, stop);
        //timeToCSV[count]= diff; 

        // deallocate timers
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        /*for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                printf("%f\t", m3[i*N+j]);
            }
            printf("\n");
        }*/

        // open a file for write
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
        free(m1);
        free(m2);
        free(m3);
        cudaFree(dev_m1);
        cudaFree(dev_m2);
        cudaFree(dev_m3);

        fclose(fp);
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
    __shared__ float ds_m1[TILE_WIDTH][TILE_WIDTH]; 
    __shared__ float ds_m2[TILE_WIDTH][TILE_WIDTH]; 

    int bx = blockIdx.x;  int by = blockIdx.y; 
    int tx = threadIdx.x; int ty = threadIdx.y; 

    int row = by * blockDim.y + ty; 
    int col = bx * blockDim.x + tx; 
    float Pvalue = 0; 

    // Loop over the M and N tiles required to compute the P element 
    for (int p = 0; p < TILE_WIDTH; ++p) { 
        // m1 and m2 load into shared memory
        // check to make sure within boundaries, if else load 0
        if(row < N && p*TILE_WIDTH+tx < N){
            ds_m1[ty][tx] = m1[row*N + p*TILE_WIDTH+tx]; 
            //printf("if\n");
        } else {
            //printf("its else\n");
            ds_m1[ty][tx] = 0;
        }

        // check to make sure within boundaries, if else load 0
        if(col < N && p*TILE_WIDTH+ty < N){
            ds_m2[ty][tx] = m2[(p*TILE_WIDTH+ty)*N + col]; 
        } else{
            ds_m2[ty][tx] = 0;
        }
        __syncthreads(); 
        
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Pvalue += ds_m1[ty][i] * ds_m2[i][tx]; 
        }
        __syncthreads(); 
    }  
    m3[row*N+col] = Pvalue; 
}

__global__ void generateRandomNumbers(curandState_t* states, float* numbers, int N) {
    // Initialize the curandState_t object for each thread
    int col = threadIdx.x + blockDim.x * blockIdx.x; 
    int row = threadIdx.y + blockDim.y * blockIdx.y; 
    int idx = row * N + col;

    // initialize curand (seed, sequence, offset, state)
    curand_init(1234, idx, 0, &states[idx]);
    
    // Generate a random number and store it in the numbers array
    if (idx < N * N){
        numbers[idx] = curand_uniform(&states[idx]);  
    }
}