#include <iostream>
#include <vector>

#include <cuda.h>
#include <vector_types.h>

#define BLUR_SIZE 16 // size of surrounding image is 2X this
#define BLOCK_DIM 16
#define TILE_WIDTH BLOCK_DIM

#include "bitmap_image.hpp"

using namespace std;

__global__ void blurKernel (uchar3 *in, uchar3 *out, int width, int height) {
    //int col = blockIdx.x * blockDim.x + threadIdx.x;
    //int row = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("in[%d][%d] = %d\n", row, col, in[row * width + col].x);

    // shared memory 
    __shared__ uchar3 sharedIn[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y; 
    int tx = threadIdx.x; int ty = threadIdx.y; 

    int row = by * blockDim.y + ty; 
    int col = bx * blockDim.x + tx; 
    
    // check to see if the current position in the input is a valid pixel 
    if (col < width && row < height) {
        // store the current pixel in the shared memory at the index of the thread in sharedIn
        sharedIn[ty][tx] = in[row*width+col];
        //printf("sharedin[%d][%d] = %u\n", ty, tx, sharedIn[ty][tx].y);
        // wait for all data 
        __syncthreads();

        int3 pixVal;
        pixVal.x = 0; pixVal.y = 0; pixVal.z = 0;
        int pixels = 0;

        // get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++) {

                int curRow = ty + blurRow;
                int curCol = tx + blurCol;

                // verify that we have a valid image pixel
                // calculate the pixVal using the shared memory
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    pixVal.x += sharedIn[curRow][curCol].x;
                    pixVal.y += sharedIn[curRow][curCol].y;
                    pixVal.z += sharedIn[curRow][curCol].z;
                    pixels++; // keep track of number of pixels in the accumulated total
                    //printf("valid: row %d, col %d", curRow, curCol);
                }
            }
        }
        __syncthreads();

        // write our new pixel value out
        out[row * width + col].x = (unsigned char)(pixVal.x / pixels);
        out[row * width + col].y = (unsigned char)(pixVal.y / pixels);
        out[row * width + col].z = (unsigned char)(pixVal.z / pixels);
    }
}

int main(int argc, char **argv){
    if (argc != 2) {
        cerr << "format: " << argv[0] << " { 24-bit BMP Image Filename }" << endl;
        exit(1);
    }
    
    bitmap_image bmp(argv[1]);

    if(!bmp){
        cerr << "Image not found" << endl;
        exit(1);
    }

    int height = bmp.height();
    int width = bmp.width();
    
    cout << "Image dimensions:" << endl;
    cout << "height: " << height << " width: " << width << endl;

    cout << "Blurring " << argv[1] << "..." << endl;

    //Transform image into vector of doubles
    vector<uchar3> input_image;
    rgb_t color;
    for(int x = 0; x < width; x++){
        for(int y = 0; y < height; y++){
            bmp.get_pixel(x, y, color);
            input_image.push_back( {color.red, color.green, color.blue} );
        }
    }

    vector<uchar3> output_image(input_image.size());

    uchar3 *d_in, *d_out;
    int img_size = (input_image.size() * sizeof(char) * 3);
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);

    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input_image.data(), img_size, cudaMemcpyHostToDevice);

    /////// ORIGINAL ///////
    //dim3 dimGrid(ceil(width / 16), ceil(height / 16), 1);
    //dim3 dimBlock(16, 16, 1);

    //////// Modified /////////
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); 
    dim3 dimGrid((int)ceil((width+BLOCK_DIM-1)/BLOCK_DIM), (int)ceil((height+BLOCK_DIM-1)/BLOCK_DIM));

    // allocate timers
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    // start timer
    cudaEventRecord(start,0);

    blurKernel<<< dimGrid , dimBlock >>> (d_in, d_out, width, height);
    cudaDeviceSynchronize();
    
    // stop timer and print time
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float diff;
    cudaEventElapsedTime(&diff, start, stop);
    //timeToCSV[count]= diff; 

    // deallocate timers
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    
    //Set updated pixels
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            int pos = x * height + y;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }

    cout << "Conversion complete." << endl;
    
    // output time 
    printf("time: %f s\n", diff/1000.0);
    
    bmp.save_image("./blurred.bmp");

    cudaFree(d_in);
    cudaFree(d_out);
}