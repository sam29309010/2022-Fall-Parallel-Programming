Parallel Programming Lab6 Report: OpenCL Programming
===

###### tags: `PP`

* Lab6 Report of CSIC30148 2022 Fall Parallel Programming @ NYCU
* Editor: 310551145 Cheng-Che Lu

### Convolution Operation Implementation in OpenCL
> Q1: Explain your implementation. How do you optimize the performance of convolution?

#### Program Overview
Like other typical OpenCL programs, the program first creates the command queue and allocates the device memory. Then, it creates the kernel, sets the kernel arguments, and specifies the workgroup size. After running the kernel, it copies the data from the device to the host and releases the OpenCL objects at the end.

#### Optimization Details
1. `CL_MEM_USE_HOST_PTR` flag is used in function `clCreateBuffer` when allocating device memory for filter and input image data.
2. Shift operation is used to replace multiplication / division of power of two.
3. The group size is set to `imageHeight * imageWidth / 4`. In this way, each work processes four pixels' convolution operations.
4. Before performing the partial summation of convolution ops, the program checks the boundary first checking first and skips those zero-valued filters for better performance.

### Convolution Operation Implementation in CUDA
> Q2: Rewrite the program using CUDA. (1) Explain your CUDA implementation, (2) plot a chart to show the performance difference between using OpenCL and CUDA, and (3) explain the result.

#### Implementation

```cpp
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
```

1. **The implementation and optimization details follow the one in openCL.**
2. Datatype `float4` is used to process four pixels' convolution operations for each thread.
3. `cudaMemcpy` is explicitly called to transfer the data between the device and host.

#### Performance Comparison

| Filter Size | OpenCL Perf. | CUDA Perf. |
| ----------- | ------------ | ---------- |
| 3x3         | 0.429        | 0.551      |
| 5x5         | 0.467        | 0.593      |
| 7x7         | 0.470        | 0.608      |

![](https://i.imgur.com/PGv7ZIt.png)


#### Performance Analysis
For 3x3, 5x5, and 7x7 filters, the program in OpenCL runs faster than the program written in CUDA. A reasonable assumption would be that different memory management methods lead to such differences. For the program in OpenCL, flag `CL_MEM_USE_HOST_PTR` enables the program directly access data in the host if possible. In contrast, for the program in CUDA, the memory transfer between the host and device is done by explicitly duplicating the data. Such implementation may be sub-optimal from the performance aspect in this case.