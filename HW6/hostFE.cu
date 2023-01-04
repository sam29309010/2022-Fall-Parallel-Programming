#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "hostFE.h"
}

__global__ void convKernel(
    const int half_filter,
    const int imageWidth,
    const int imageHeight,
    const float *d_filter,
    const float *d_in_img,
    float* d_out_img
)
{
    // Should assure the imageWidth is divisible by 4

    const int center_width = (blockIdx.x * blockDim.x + threadIdx.x) << 2; 
    const int center_height = blockIdx.y * blockDim.y + threadIdx.y;
    const int filter_size = half_filter << 1 + 1;

    if ((center_width >= imageWidth) || (center_height >= imageHeight))
        return; 
    
    float4 sums = make_float4(0.0, 0.0, 0.0, 0.0);
    int filter_idx = 0;
    
    for (int ker_height_offset = -half_filter; ker_height_offset <= half_filter; ker_height_offset++)
    {
        const int height = center_height + ker_height_offset;
        if ((height < 0) || (height >= imageHeight))
        {
          filter_idx += filter_size;
          continue;
        }
        
        const int height_idx = height * imageWidth;
        for (int ker_width_offset = -half_filter; ker_width_offset <= half_filter; ker_width_offset++, filter_idx++)
        {
            if (d_filter[filter_idx] != 0)
            {
                const int width = center_width + ker_width_offset;
                if ((width < 0) || (width >= imageWidth))
                    continue;

                const int idx = height_idx + width;
                const float4 inputs = make_float4(
                    d_in_img[idx+0],
                    d_in_img[idx+1],
                    d_in_img[idx+2],
                    d_in_img[idx+3]
                );
                const float filter_val = d_filter[filter_idx];

                sums.x += inputs.x * filter_val;
                sums.y += inputs.y * filter_val;
                sums.z += inputs.z * filter_val;
                sums.w += inputs.w * filter_val;
            }
        }
    }
    const int out_start_idx = center_height * imageWidth + center_width;
    d_out_img[out_start_idx+0] = sums.x;
    d_out_img[out_start_idx+1] = sums.y;
    d_out_img[out_start_idx+2] = sums.z;
    d_out_img[out_start_idx+3] = sums.w;
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int blocksX = (int) ceil(imageWidth / 16.0);
    int blocksY = (int) ceil(imageHeight / 16.0);

    dim3 threadsPerBlock(4, 16), blocksPerGrid(blocksX, blocksY);

    const int half_filter = filterWidth >> 1;
    const int filter_size = filterWidth * filterWidth * sizeof(float);
    const int image_size = imageWidth * imageHeight * sizeof(float);

    float *d_filter, *d_in_img, *d_out_img;

    cudaMalloc((void **)&d_filter, filter_size);
    cudaMalloc((void **)&d_in_img, image_size);
    cudaMalloc((void **)&d_out_img, image_size);
    cudaMemcpy(d_in_img, inputImage, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filter_size, cudaMemcpyHostToDevice);

    convKernel <<<blocksPerGrid, threadsPerBlock>>> (half_filter, imageWidth, imageHeight, d_filter, d_in_img, d_out_img);

    cudaMemcpy(outputImage, d_out_img, image_size, cudaMemcpyDeviceToHost);
    cudaFree(d_filter);
    cudaFree(d_in_img);
    cudaFree(d_out_img);
}