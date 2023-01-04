#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8
#define NUM_STREAM 10

__global__ void mandelKernel(int *output, float lowerX, float lowerY, float stepX, float stepY, int resX, int offset, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x; 
    int thisY = blockIdx.y * blockDim.y + threadIdx.y + offset; 

    int index = thisY * resX + thisX; // (j * width + i);

    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;
    float z_re = c_re;
    float z_im = c_im;
    
    int iter = 0;
    for (; iter < maxIterations; ++iter)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    
    output[index] = iter;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    const int img_size_mem = resX * resY * sizeof(int);
    
    // Allocate device memory
    int *dev_img;
    cudaMalloc(&dev_img, img_size_mem);
    cudaHostRegister(img, img_size_mem, cudaHostRegisterPortable);

    cudaStream_t streams[NUM_STREAM];
    for (int i = 0; i < NUM_STREAM; i++)
        cudaStreamCreate(&streams[i]);

    int grid_step = resY / NUM_STREAM;
    int grid_size = img_size_mem / NUM_STREAM;

    // Invoke kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(resX / BLOCK_SIZE, grid_step / BLOCK_SIZE);
    
    int offset = 0;
    for (int i = 0; i < NUM_STREAM; i++)
    {
        mandelKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(dev_img, lowerX, lowerY, stepX, stepY, resX, offset, maxIterations);
        cudaMemcpyAsync(img + resX * offset, dev_img + resX * offset, grid_size, cudaMemcpyDeviceToHost, streams[i]);
        offset += grid_step;
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < NUM_STREAM; i++)
        cudaStreamDestroy(streams[i]);
    cudaHostUnregister(img);
    cudaFree(dev_img);
}