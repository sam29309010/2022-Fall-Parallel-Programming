Parallel Programming Lab2 Report: Multi-thread Programming
===

###### tags: `PP`

* Lab2 Report of CSIC30148 2022 Fall Parallel Programming @ NYCU
* Editor: 310551145 Cheng-Che Lu

### TL;DR
*Pthread* provides thread-level management functions. Load balance issue, hardware support, reentrant property, and thread-safe property should be considered when using *pthread* to parallelize the program. 

### Performance improvement of naive multi-thread optimization
> Is speedup linear in the number of threads used? In your writeup hypothesize why this is (or is not) the case?

A linear performance improvement along the number of used threads is not observed, especially in view one with three threads. The program's elapsed time below is tested on an Intel i7-870 processor, normalized to one-thread performance. A reasonable assumption is that there may be a **workload imbalance issue**. Concretely, the function `mandelbrotSerial` iteratively calls the function `mandel` for each pixel value of the output `.ppm` image, whose computation time mainly depends on the total number of the member that belongs to the Mandelbrot set. If there's a spatial correlation in counting Mandelbrot set members, allocating a chunk of contiguous data to each thread would be a problem.

| View     | # Thread | Speedup  |
| -------- | -------- | -------- |
| 1        | 2        | 1.91x    |
| 1        | 3        | 1.62x    |
| 1        | 4        | 2.32x    |
| 2        | 2        | 1.65x    |
| 2        | 3        | 2.07x    |
| 2        | 4        | 2.49x    |

```cpp
for (i = 0; i < count; ++i)
{
    if (z_re * z_re + z_im * z_im > 4.f)
        break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
}
```

> How do your measurements explain the speedup graph you previously created?

The table below shows **a wide difference in average execution time (ms) across threads** through time profiling. This problem is most significant for view one with three threads, which is similar to the speedup result.

| View | # Thread | Thread 0 | Thread 1 | Thread 2 | Thread 3 |
| ---- | -------- | -------- | -------- | -------- | -------- |
| 1    | 2        | 239.95   | 241.78   |          |          |
| 1    | 3        | 103.31   | 286.29   | 103.51   |          |
| 1    | 4        | 49.33    | 195.76   | 195.68   | 49.73    |
| 2    | 2        | 165.44   | 118.32   |          |          |
| 2    | 3        | 134.74   | 90.20    | 79.57    |          |
| 2    | 4        | 110.86   | 66.04    | 67.08    | 65.73    |

For further validation, the report also compares the summation of the processed pixel value. Both metrics show the same distribution, which illustrates that workload imbalance indeed is the cause of such performance degradation.

| View | Total thread used | Thread 0 | Thread 1 | Thread 2 | Thread 3 |
| ---- | ----------------- | -------- | -------- | -------- | -------- |
| 1    | 2                 | 84,344K  | 84,628K  |          |          |
| 1    | 3                 | 39,271K  | 90,281K  | 39,421K  |          |
| 1    | 4                 | 22,787K  | 61,557K  | 61,727K  | 22,901K  |
| 2    | 2                 | 96,997K  | 82,934K  |          |          |
| 2    | 3                 | 68,648K  | 57,297K  | 53,986K  |          |
| 2    | 4                 | 54,471K  | 42,527K  | 42,659K  | 40,275K  |



### Multi-thread optimization with spatial interleaving
>In your write-up, describe your approach to parallelization and report the final 4-thread speedup obtained.

An intuitive method to resolve the workload imbalance problem is to allocate the computation tasks more uniformly. This could be done by **cyclic partition**, which assigns the components (i.e., one row of processed data) in a round-robin fashion. As the code is shown below, the *x*th row of data will be assigned to the *x%numThreads*th thread.

```cpp
for (int row=args->threadId; row<(int) args->height; row+=args->numThreads){
    mandelbrotSerial(
        args->x0, args->y0, args->x1, args->y1,
        args->width, args->height,
        row, 1,
        args->maxIterations,
        args->output
    );
}
```

| View | Total thread used | Speedup  | Thread 0 | Thread 1 | Thread 2 | Thread 3 |
| ---- | ----------------- | -------- | -------- | -------- | -------- | -------- |
| 1    | 2                 | 1.92x    | 84,511K  | 84,462K  |          |          |
| 1    | 3                 | 2.66x    | 56,369K  | 56,302K  | 56,302K  |          |
| 1    | 4                 | 3.67x    | 42,272K  | 42,231K  | 42,239K  | 42,231K  |
| 2    | 2                 | 1.91x    | 89,990K  | 89,941K  |          |          |
| 2    | 3                 | 2.67x    | 60,007K  | 59,979K  | 59,946K  |          |
| 2    | 4                 | 3.44x    | 45,024K  | 44,989K  | 44,966K  | 44,952K  |


### Limit of Hardware Parallism
>Now run your improved code with eight threads. Is performance noticeably greater than when running with four threads? Why or why not?

The table shows the performance improvement from two threads used to eight threads used. There is a linear improvement only until four-threaded parallelism. The main reason is that the processor here provides four cores and four threads. Specifying further parallelism requires more **context switch operation** while with the same thread-parallelism in hardware. Thus, it even leads to poorer performance.

| View | 2 Threads | 3 Threads | 4 Threads | 5 Threads | 6 Threads | 7 Threads | 8 Threads |
| ---- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 1    | 1.92x     | 2.66x     | 3.57x     | 2.96x     | 3.29x     | 3.46x     | 3.46x     |
| 2    | 1.91x     | 2.67x     | 3.55x     | 3.01x     | 3.30x     | 3.40x     | 3.33x     |


### Implementation of Parallel PI Estimation Using Pthreads
Simply modifying the serial implementation into a multi-thread one may lead to incorrect result since *srand* and *rand* function is not guaranteed to be thread-safe, as shown in the reference. One alternative would be using a thread-safe version of *rand* called *rand_r*. SIMD-instructions-based pseudo-random number generators (RNG) method such as *SIMDxorshift* would be a decent choice to pursue better performance further.

### Reference
1. [C *srand()* function](https://en.cppreference.com/w/c/numeric/random/srand)
2. [C *rand()* function](https://en.cppreference.com/w/c/numeric/random/rand)
3. [SIMDxorshift: SIMD pseudo-random number generators](https://github.com/lemire/SIMDxorshift)
4. [Random number generators for C++ performance tested](https://thompsonsed.co.uk/random-number-generators-for-c-performance-tested)

