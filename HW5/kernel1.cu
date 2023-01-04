#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8

__global__ void mandelKernel(int *output, float lowerX, float lowerY, float stepX, float stepY, int resX, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x; 
    int thisY = blockIdx.y * blockDim.y + threadIdx.y; 

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
    int *hst_img = (int *) malloc(img_size_mem);
    
    // Allocate device memory
    int *dev_img;
    cudaMalloc(&dev_img, img_size_mem);

    // Copy data from host memory to device memory
    cudaMemcpy(dev_img, hst_img, img_size_mem, cudaMemcpyHostToDevice);

    // Invoke kernel
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(resX / BLOCK_SIZE, resY / BLOCK_SIZE);

    mandelKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_img, lowerX, lowerY, stepX, stepY, resX, maxIterations);

    cudaMemcpy(hst_img, dev_img, img_size_mem, cudaMemcpyDeviceToHost);
    memcpy(img, hst_img, img_size_mem);

    free(hst_img);
    cudaFree(dev_img);
}
