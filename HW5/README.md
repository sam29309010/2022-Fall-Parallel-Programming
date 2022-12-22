Parallel Programming Lab5 Report: CUDA Programming
===

###### tags: `PP`

* Lab5 Report of CSIC30148 2022 Fall Parallel Programming @ NYCU
* Editor: 310551145 Cheng-Che Lu

### TL;DR
CUDA provides a parallel computing API that allows the software to accelerate the program using GPU. To fully exploit GPU's computation resource, the programmer should consider memory communication (between host and device), data locality, and other parallelism mechanisms, such as streaming and asynchronous operations.

### Method Comparison
> Q1 What are the pros and cons of the three methods? Give an assumption about their performances.

#### Method 1
- Pros: Each thread deals with the computation of only one pixel, so the **process parallelism is relatively high**.
- Cons: There may be a **load balance issue** since the number of required operations of each pixel could vary greatly.

#### Method 2
- Pros: It could be **more efficient when transferring data** if the given data is in 2d format and aligned.
- Cons: There may exist **extra usage of memory space** to align the data in host and device memory if each "row" of data is less than 256 bits.

#### Method 3
- Pros: It could **save the computation resource** since the computation of multiple pixels is now assigned to one single thread.
- Cons: The workload is heavier now for each thread, so **performance may be poorer** than previous methods.

**Based on the properties listed above, it is reasonable to assume that method 3 performs slower than two other methods, and methods 1 & 2 would perform with just a slight difference.**

### Experimental Result
> Q2 How are the performances of the three methods? Plot a chart to show the differences among the three methods.

| Method   | View     | 1K Iter. | 10K Iter. | 100K Iter. |
| -------- | -------- | -------- | --------- | ---------- |
| Method 1 | View 1   | 7.17 ms  | 34.54 ms  | 306.34 ms  |
| Method 1 | View 2   | 4.97 ms  | 7.32 ms   | 29.32 ms   |
| Method 2 | View 1   | 9.03 ms  | 35.94 ms  | 308.09 ms  |
| Method 2 | View 2   | 6.09 ms  | 8.38 ms   | 30.45 ms   |
| Method 3 | View 1   | 10.09 ms | 39.17 ms  | 340.52 ms  |
| Method 3 | View 2   | 6.40 ms  | 9.67 ms   | 45.10 ms   |

![](https://i.imgur.com/2c5chwB.png)
![](https://i.imgur.com/Kl5yChX.png)



### Performance Analysis
> Q3 Explain the performance differences thoroughly based on your experimental results. Does the results match your assumption? Why or why not.

As shown above, method 3 performs significantly slower than the two other methods. **It is mainly because of the heavier workload for each thread using method 3** (from one pixel to multiple pixels' computation). Increased computation leads to longer running time. Also, **such a problem would get worse if those heavy-computation pixels fell into the same thread**. On the other hand, **method 1 and 2 performs almost the same from a time aspect, mainly because of their similar required computation for each thread.**

**Such deduction could be verified through *nvprof* profiling**. Below are the major GPU activities of methods 1 to 3.

```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   46.21%  2.98210s        10  298.21ms  296.45ms  299.36ms  mandelKernel(int*, float, float, float, float, int, int)
                   46.87%  2.98360s        10  298.36ms  297.74ms  299.38ms  mandelKernel(int*, float, float, float, float, int, int)
                   49.56%  3.32823s        10  332.82ms  331.34ms  334.76ms  mandelKernel(int*, float, float, float, float, int, int)
```


### Other optimization
> Q4 Can we do even better? Think a better approach and explain it. Implement your method in kernel4.cu.

There are two modifications (based on method 1) for further optimization. First, the program removes the unnecessary host memory allocation. Second, CUDA Streaming & asynchronous memory copy are adopted enhance the parallelism (by overlapping the numerical computation and data transferring operation).

| Method   | View     | 1K Iter. | 10K Iter. | 100K Iter. |
| -------- | -------- | -------- | --------- | ---------- |
| Method 4 | View 1   | 3.85 ms  | 29.13 ms  | 280.46 ms  |
| Method 4 | View 2   | 2.03 ms  | 3.85 ms   | 24.61 ms   |


### Reference
1. [CUDA Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)