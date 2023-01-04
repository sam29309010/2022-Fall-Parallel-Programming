#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    // cl_int status;
    const int half_filter = filterWidth >> 1;
    const int filter_size = filterWidth * filterWidth * sizeof(float);
    const int image_size = imageWidth * imageHeight * sizeof(float);

    // create command queue
    cl_command_queue cmd_q = clCreateCommandQueue(*context, *device, 0, NULL);

    // allocate device memory
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filter_size, filter, NULL); // CL_MEM_COPY_HOST_PTR
    cl_mem d_in_img = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, image_size, inputImage, NULL);
    cl_mem d_out_img = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_size, NULL, NULL);

    // create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    // Set arguments for the kernel
    clSetKernelArg(kernel, 0, sizeof(int), &half_filter);
    clSetKernelArg(kernel, 1, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 2, sizeof(int), &imageHeight);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_filter);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_in_img);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_out_img);

    // set workgroups sizes
    size_t global_work_size = (imageWidth * imageHeight) >> 2;

    // run kernel
    clEnqueueNDRangeKernel(cmd_q, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

    // copy data from device to host
    clEnqueueReadBuffer(cmd_q, d_out_img, CL_TRUE, 0, image_size, outputImage, 0, NULL, NULL);

    // // release opencl object
    // // removed for better performance
    // clReleaseCommandQueue(cmd_q);
    // clReleaseMemObject(d_filter);
    // clReleaseMemObject(d_in_img);
    // clReleaseMemObject(d_out_img);
    // clReleaseKernel(kernel);
}