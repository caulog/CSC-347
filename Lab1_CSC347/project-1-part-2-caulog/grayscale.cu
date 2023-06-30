/*
* Olivia Caulfield
* Cho
* CSC 347
* 2/2/23
*/

#include <iostream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>
#include "bitmap_image.hpp"
#define N 1024
#define BLOCK_DIM 16

using namespace std;

__global__ void color_to_grey(uchar3 *input_image, uchar3 *output_image, int width, int height)
{
    // calculate row and column of current element
    int col = blockDim.x * blockIdx.x + threadIdx.x;                    
    int row = blockDim.y * blockIdx.y + threadIdx.y;                    

    // if the element is within the image
    if (col < width && row < height){
        // map to a 1D offset
        //int index = col + row * height;
        int index = width * col + row;
        
        // calculate RGB for grey conversion
        float R = input_image[index].x * 0.299;
        float G = input_image[index].y * 0.578;
        float B = input_image[index].z * 0.114;

        // convert image to grey
        output_image[index].x = R+B+G;
        output_image[index].y = R+B+G;
        output_image[index].z = R+B+G;

    }
}


int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "format: " << argv[0] << " { 24-bit BMP Image Filename }" << endl;
        exit(1);
    }
    
    bitmap_image bmp(argv[1]);

    if(!bmp)
    {
        cerr << "Image not found" << endl;
        exit(1);
    }

    int height = bmp.height();
    int width = bmp.width();
    
    cout << "Image dimensions:" << endl;
    cout << "height: " << height << " width: " << width << endl;

    cout << "Converting " << argv[1] << " from color to grayscale..." << endl;


    //Transform image into vector of doubles
    vector<uchar3> input_image;
    rgb_t color;
    for(int x = 0; x < height; x++)
    {
        for(int y = 0; y < width; y++)
        {
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

    // set block and grid dim
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); 
    dim3 dimGrid((int)ceil(width/BLOCK_DIM), (int)ceil(height/BLOCK_DIM));

    color_to_grey<<< dimGrid , dimBlock >>> (d_in, d_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    
    //Set updated pixels
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            int pos = x * width + y;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }

    cout << "Conversion complete." << endl;
    
    bmp.save_image("./grayscaled.bmp");

    cudaFree(d_in);
    cudaFree(d_out);
}
