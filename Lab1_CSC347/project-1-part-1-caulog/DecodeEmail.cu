/* 
**  Olivia Caulfield
**  Cho
**  CSC 347
**  2/2/23 
*/

#include <stdio.h> 
#include <stdlib.h>

__global__ void decode (char *encoded, char *decoded, int len);

int main(int argc, char **argv){
   // Initialize variables
   // File pointer to message for CPU memory
    FILE *message;
    // GPU memory
    char *dev_encoded, *dev_decoded; 
    // character read
    char c;

    // check for valid number of command line arguments
    // if all is good open file, if file is not found, print error
    if(argv[1] == NULL || argv[2] != NULL){
        printf("Invalid number of command line arguements\n");
        exit(100); 
    }
    message = fopen(argv[1], "r");
    if(message == NULL){
        printf("File '%s' not found\n", argv[1]);
        exit (100);
    }

    // find file size in char
    int i = 0;
    c = fgetc(message);
    while (c != EOF){
       c = fgetc(message);
       i++;
    }    

    // CPU memory size of char length
    int N = i;
    char encoded[N], decoded[N];

    // store the file in a char pointer
    rewind(message);
    i = 0;
    c = fgetc(message);
    while (c != EOF){
       encoded[i] = c;
       c = fgetc(message);
       i++;
    }
    // manually add the character encoded[i+1] = '\0';
    encoded[i] = '\0';    
    fclose(message);

    int size = N * sizeof(char); 
    
    // allocate GPU memory
    cudaMalloc((void**)&dev_encoded, size); 
    cudaMalloc((void**)&dev_decoded, size); 

    // copy CPU memory to GPU
    cudaMemcpy(dev_encoded, encoded, size, cudaMemcpyHostToDevice); 

    // call global kernel function
    decode<<<1,N>>>(dev_encoded, dev_decoded, N); 
    // waits for all threads to finish --> returns an error if one of the preceding tasks has failed.
    cudaDeviceSynchronize(); 

    // copy the GPU memory back to the CPU
    cudaMemcpy(decoded, dev_decoded, size, cudaMemcpyDeviceToHost); 
    
    // free the GPU memory
    cudaFree(dev_encoded); 
    cudaFree(dev_decoded);

    // print the decoded message
    int j = 0;
    while (j < N){
        printf("%c", decoded[j]);
        j++;
    }
    printf("\n");

    // exit
    exit (0);
}

// decode kernel function
__global__ void decode (char *encoded, char *decoded, int len){
    int i = threadIdx.x; 

    // as long as not end of string character, decode.
    if (encoded[i] != '\0'){
        decoded[i] = encoded[i]-1; 
    } 
    if (i == len-2){
        decoded[i] = '\0';
    }
}



