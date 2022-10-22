Parallel Programming Lab1 Report: SIMD Programming
===

###### tags: `PP`

* Lab1 Report of CSIC30148 2022 Fall Parallel Programming @ NYCU
* Editor: 310551145 Cheng-Che Lu

### TL;DR
SIMD Programming exploits instruction-level prarllism. Common optimization includes loop-vectorization and advanced SIMD instructions such as AVX2 or AVX512. Better data alignment and variable aliases could further improve the performance. 

`loop-vectorization`, `__restrict`, `__builtin_assume_aligned`, `AVX2`

### Relation between vector utilization and vector width
> Q1-1: Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

There is a negative correlation between vector width and vector utilization, as shown in the table below, which records the usage of instruction across different vector widths tested on an array with 10000+16 independent elements.

| Vector Width | Maximum Exponent | # Vector Inst. | Vector Utilization |
| ------------ | ---------------- | -------------------- | ------------------ |
| 2     | 10 | 172,469     | 82.7%     |
| 4     | 10 | 102,589     | 75.9%     |
| 8     | 10 | 58,019      | 71.7%     |
| 16    | 10 | 31,344      | 69.4%     |

To identify the cause of such findings, the following table further breakdowns the vector utilization by each instruction type. Each entry shows the instruction count and its average utilization.

| Instruction Type | VECTOR_WDITH = 2 | VECTOR_WDITH = 4 | VECTOR_WDITH = 8 | VECTOR_WIDTH = 16 |
| ---------------- | ---------------- | ---------------- | ---------------- | ----------------- |
| cntbits | 26,493 / 100%        | 16,517 / 100%        | 9,603 / 100%        | 5,268 / 100%    |
| masknot | 5,000 / 100%         | 2,500 / 100%         | 1,250 / 100%        | 625 / 100%      |
| veq     | 5,000 / 100%         | 2,500 / 100%         | 1,250 / 100%        | 625 / 100%      |
| vgt     | 31,493 / 100%        | 19,017 / 100%        | 10,853 / 100%       | 5,893 / 100%    |
| vload   | 10,000 / 100%        | 5,000 / 100%         | 2,500 / 100%        | 1,250 / 100%    |
| vlt     | 26,493 / 55.85%      | 16,517 / 44.79%      | 9,603 / 38.52%      | 5,268 / 35.11%  |
| vmove   | 10,000 / 57.67%      | 5,000 / 57.67%       | 2,500 / 57.67%      | 1,250 / 57.67%  |
| vmult   | 21,493 / 68.84%      | 14,017 / 52.78%      | 8,353 / 44.28%      | 4,643 / 39.83%  |
| vset    | 5,004 / 100%         | 2,504 / 100%         | 1,254 / 100%        | 629 / 100%      |
| vstore  | 5,000 / 100%         | 2,500 / 100%         | 1,250 / 100%        | 625 / 100%      |
| vsub    | 26,493 / 72.9%       | 16,517 / 58.47%      | 9,603 / 50.28%      | 5,268 / 45.83%  |

Instructions *vlt*, *vmult*, and *vsub* account for a large proportion of the whole vector instruction usage, with lower vector utilization as the vector width increases. Since the program calls these instructions mainly within a while loop for exponent operation, a reasonable explanation for lower utilization would be the multiplication operation with sparse input. Concretely, the maximum number of exponent within the sub-vector and clamping condition determine the total number of instructions of clamped exponent operation for each sub-vector. As the vector width increases, there's a higher chance that the maximum exponent would increase too. Hence, it leads to more computation cycles for a few elements with large exponent while leaving computation for other elements idle.

To illustrate such a situation, the table below examines the effect of maximum exponent for vector utilization. Not surprisingly, a larger maximum exponent aggravates this low-utilization problem.

| Vector Width | Maximum Exponent | # Vector Inst. | Vector Utilization |
| ------------ | ---------------- | -------------------- | ------------------ |
| 16     | 10  | 31,344      | 69.4%     |
| 16     | 25  | 72,804      | 64.2%     |
| 16     | 50  | 142,304     | 61.4%     |
| 16     | 100 | 282,449     | 59.7%     |

An unaligned array may lower the utilization. While assuming the array length is large enough, this effect would be trivial.
Also, [a more compact algorithm](https://eli.thegreenplace.net/2009/03/21/efficient-integer-exponentiation-algorithms) for the exponent operation could alleviate this utilization issue when exponent gets larger. Instead of performing multiplication incrementally (adding 1x of base number each time), exponentiation by squaring could reduce the time complexity to *log(maximum_exp)* and 50% utilization on average.


### Specifying data alignment for AVX2 instructions 
> Q2-1: Fix the code to make sure it uses aligned moves for the best performance.

According to [Intel® C++ Compiler Classic Developer Guide and Reference](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-avx2.html), AVX2 extends AVX by promoting most 128-bit SIMD integer instructions with 256-bit numeric processing capabilities.

The compiler assumes no alignment of given processed data by default to ensure correctness for any input, thus compiling with MOV**U**PS-based (Move **Unaligned** Packed Single-Precision Floating-Point Values) instructions. To explicitly specify alignment for AVX2 instructions, the processed data should be aligned by 256bits (32bytes), as shown below.

```cpp
a = (float *)__builtin_assume_aligned(a, 32);
b = (float *)__builtin_assume_aligned(b, 32);
c = (float *)__builtin_assume_aligned(c, 32);
```


This way, MOV**A**PS-based (Move Aligned Packed Single-Precision Floating-Point Values) instructions will be called.

Original:
```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovups	(%rbx,%rcx,4), %ymm0
	vmovups	32(%rbx,%rcx,4), %ymm1
	vmovups	64(%rbx,%rcx,4), %ymm2
	vmovups	96(%rbx,%rcx,4), %ymm3
	vaddps	(%r15,%rcx,4), %ymm0, %ymm0
	vaddps	32(%r15,%rcx,4), %ymm1, %ymm1
	vaddps	64(%r15,%rcx,4), %ymm2, %ymm2
	vaddps	96(%r15,%rcx,4), %ymm3, %ymm3
	vmovups	%ymm0, (%r14,%rcx,4)
	vmovups	%ymm1, 32(%r14,%rcx,4)
	vmovups	%ymm2, 64(%r14,%rcx,4)
	vmovups	%ymm3, 96(%r14,%rcx,4)
	addq	$32, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_3
```

Modified:
```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovaps	(%rbx,%rcx,4), %ymm0
	vmovaps	32(%rbx,%rcx,4), %ymm1
	vmovaps	64(%rbx,%rcx,4), %ymm2
	vmovaps	96(%rbx,%rcx,4), %ymm3
	vaddps	(%r15,%rcx,4), %ymm0, %ymm0
	vaddps	32(%r15,%rcx,4), %ymm1, %ymm1
	vaddps	64(%r15,%rcx,4), %ymm2, %ymm2
	vaddps	96(%r15,%rcx,4), %ymm3, %ymm3
	vmovaps	%ymm0, (%r14,%rcx,4)
	vmovaps	%ymm1, 32(%r14,%rcx,4)
	vmovaps	%ymm2, 64(%r14,%rcx,4)
	vmovaps	%ymm3, 96(%r14,%rcx,4)
	addq	$32, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_3
```

Besides, the whole optimization details for *test1*, *test2* and, *test3* functions (vector add, max, and sum operation, respectively) are listed here.

#### Vector Add Operation
##### Unvectorized Implementation
```MOVSS``` (Move or Merge Scalar Single-Precision Floating-Point Value) and ```ADDSS``` (Add Scalar Single-Precision Floating-Point Values) instructions are called without exploiting any parallelism.
```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%rbx,%rcx,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	addss	(%r15,%rcx,4), %xmm0
	movss	%xmm0, (%r14,%rcx,4)
	movss	4(%rbx,%rcx,4), %xmm0   # xmm0 = mem[0],zero,zero,zero
	addss	4(%r15,%rcx,4), %xmm0
	movss	%xmm0, 4(%r14,%rcx,4)
	movss	8(%rbx,%rcx,4), %xmm0   # xmm0 = mem[0],zero,zero,zero
	addss	8(%r15,%rcx,4), %xmm0
	movss	%xmm0, 8(%r14,%rcx,4)
	movss	12(%rbx,%rcx,4), %xmm0  # xmm0 = mem[0],zero,zero,zero
	addss	12(%r15,%rcx,4), %xmm0
	movss	%xmm0, 12(%r14,%rcx,4)
	addq	$4, %rcx
	cmpq	$1024, %rcx             # imm = 0x400
	jne	.LBB0_3
```

##### Vectorized Optimization
With flag ```-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize```, ```MOVUPS``` (Move Unaligned Packed Single-Precision Floating-Point Values) and ```ADDPS``` (Add Packed Single-Precision Floating-Point Values) instructions are called to apply loop vectorization technique. Also, ```__restrict``` keyword is used to indicate that a symbol isn't aliased in the current scope (the memory of variable is not overlapped).

```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movups	(%rbx,%rdx,4), %xmm0
	movups	16(%rbx,%rdx,4), %xmm1
	movups	(%r15,%rdx,4), %xmm2
	addps	%xmm0, %xmm2
	movups	16(%r15,%rdx,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, (%r14,%rdx,4)
	movups	%xmm0, 16(%r14,%rdx,4)
	movups	32(%rbx,%rdx,4), %xmm0
	movups	48(%rbx,%rdx,4), %xmm1
	movups	32(%r15,%rdx,4), %xmm2
	addps	%xmm0, %xmm2
	movups	48(%r15,%rdx,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, 32(%r14,%rdx,4)
	movups	%xmm0, 48(%r14,%rdx,4)
	addq	$16, %rdx
	cmpq	$1024, %rdx             # imm = 0x400
	jne	.LBB0_3
	jmp	.LBB0_4
```

##### Specifying data alignment
By explicitly specifying data alignment using ```__builtin_assume_aligned```, ```MOVAPS``` (Move Aligned Packed Single-Precision Floating-Point Values) instruction is called.

```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movaps	(%rbx,%rcx,4), %xmm0
	movaps	16(%rbx,%rcx,4), %xmm1
	addps	(%r15,%rcx,4), %xmm0
	addps	16(%r15,%rcx,4), %xmm1
	movaps	%xmm0, (%r14,%rcx,4)
	movaps	%xmm1, 16(%r14,%rcx,4)
	movaps	32(%rbx,%rcx,4), %xmm0
	movaps	48(%rbx,%rcx,4), %xmm1
	addps	32(%r15,%rcx,4), %xmm0
	addps	48(%r15,%rcx,4), %xmm1
	movaps	%xmm0, 32(%r14,%rcx,4)
	movaps	%xmm1, 48(%r14,%rcx,4)
	addq	$16, %rcx
	cmpq	$1024, %rcx             # imm = 0x400
	jne	.LBB0_3
```

##### AVX2 Instructions
With flag ```-mavx2```, ```VMOVUPS``` is called with more ILP. Combined with the adequate alignment adjustment mentioned above, ```VMOVAPS``` will replace it.

```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovaps	(%rbx,%rcx,4), %ymm0
	vmovaps	32(%rbx,%rcx,4), %ymm1
	vmovaps	64(%rbx,%rcx,4), %ymm2
	vmovaps	96(%rbx,%rcx,4), %ymm3
	vaddps	(%r15,%rcx,4), %ymm0, %ymm0
	vaddps	32(%r15,%rcx,4), %ymm1, %ymm1
	vaddps	64(%r15,%rcx,4), %ymm2, %ymm2
	vaddps	96(%r15,%rcx,4), %ymm3, %ymm3
	vmovaps	%ymm0, (%r14,%rcx,4)
	vmovaps	%ymm1, 32(%r14,%rcx,4)
	vmovaps	%ymm2, 64(%r14,%rcx,4)
	vmovaps	%ymm3, 96(%r14,%rcx,4)
	addq	$32, %rcx
	cmpq	$1024, %rcx             # imm = 0x400
	jne	.LBB0_3
```

#### Vector Max Operation
The same optimization applies to vector max operation **only when** those formatted branch operations are compiler-aware (jus as Q2-3 has mentioned).

#### Vector Sum Operation
While associative property holds for addition operation theoretical, some precision loss in practical computation may exist. To ensure the correctness of program, the compiler sequentially performs the summation by default. With flag ```-ffast-math```, reordering is allowed, and the operation could be performed faster by breaking default IEEE compliance. See more details in reference.

Original:
```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	addsd	(%rbx,%rcx,8), %xmm0
	addsd	8(%rbx,%rcx,8), %xmm0
	addsd	16(%rbx,%rcx,8), %xmm0
	addsd	24(%rbx,%rcx,8), %xmm0
	addsd	32(%rbx,%rcx,8), %xmm0
	addsd	40(%rbx,%rcx,8), %xmm0
	addsd	48(%rbx,%rcx,8), %xmm0
	addsd	56(%rbx,%rcx,8), %xmm0
	addq	$8, %rcx
	cmpq	$1024, %rcx             # imm = 0x400
	jne	.LBB0_3
```

Modified:
```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	addpd	(%rbx,%rcx,8), %xmm1
	addpd	16(%rbx,%rcx,8), %xmm0
	addpd	32(%rbx,%rcx,8), %xmm1
	addpd	48(%rbx,%rcx,8), %xmm0
	addpd	64(%rbx,%rcx,8), %xmm1
	addpd	80(%rbx,%rcx,8), %xmm0
	addpd	96(%rbx,%rcx,8), %xmm1
	addpd	112(%rbx,%rcx,8), %xmm0
	addq	$16, %rcx
	cmpq	$1024, %rcx             # imm = 0x400
	jne	.LBB0_3
```
```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vaddpd	(%rbx,%rcx,8), %ymm0, %ymm0
	vaddpd	32(%rbx,%rcx,8), %ymm1, %ymm1
	vaddpd	64(%rbx,%rcx,8), %ymm2, %ymm2
	vaddpd	96(%rbx,%rcx,8), %ymm3, %ymm3
	vaddpd	128(%rbx,%rcx,8), %ymm0, %ymm0
	vaddpd	160(%rbx,%rcx,8), %ymm1, %ymm1
	vaddpd	192(%rbx,%rcx,8), %ymm2, %ymm2
	vaddpd	224(%rbx,%rcx,8), %ymm3, %ymm3
	addq	$32, %rcx
	cmpq	$1024, %rcx             # imm = 0x400
	jne	.LBB0_3
```

As the code has shown, ```ADDSD``` (Add Scalar Double-Precision Floating-Point Values) is replaced by ```ADDPD``` (Add Packed Double-Precision Floating-Point Values) or ````VADDPD````.

### Performance analysis of parallel optimization
> Q2-2: What speedup does the vectorized code achieve over the unvectorized code? What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)? You may wish to run this experiment several times and take median elapsed times; you can report answers to the nearest 100% (e.g., 2×, 3×, etc). What can you infer about the bit width of the default vector registers on the PP machines? What about the bit width of the AVX2 vector registers.

The table shows the performance improvement for *test1*, *test2*, and *test3* functions using loop-vectorization and avx2 optimization, normalized with respect to the unvectorized one. Elapsed time is estimated by repeating the experiments ten times and taking an average of it on the PC with AMD Ryzen 5-5600X CPU.

Compared to unvectorized implementation, vectorized one w.o/w AVX2 optimization could achieve 3.52x-4.02x / 7.10x-14.09x times better performance.

Script:
```bash
# estimate_time.sh
for i in {1..10}
do
    ./test_auto_vectorize $@
done

# command
time ./estimate_time.sh -t {1,2,3}
```

Vector Add Operation:
| Optimization | Avg. Time | Normalized Performance |
| ------------ | --------- | ---------------------- |
| Unvectorized implementation    | 5.0176s | 1x |
| + Loop-vectorized optimization | 1.4267s | 3.52x |
| + AVX2 optimization            | 0.7065s | 7.10x |

Vector Max Operation:
| Optimization | Avg. Time | Normalized Performance |
| ------------ | --------- | ---------------------- |
| Unvectorized implementation    | 5.5620s | 1x |
| + Loop-vectorized optimization | 1.3846s | 4.02x |
| + AVX2 optimization            | 0.7126s | 7.81x |

Vector Sum Operation:
| Optimization | Avg. Time | Normalized Performance |
| ------------ | --------- | ---------------------- |
| Unvectorized implementation    | 13.2584s | 1x |
| + Loop-vectorized optimization | 3.3729s | 3.93x |
| + AVX2 optimization            | 0.9413s | 14.09x |

Take *test1* function as an example. The assembly code of computation part w vectorized optimization & w / w.o AVX2 are as follow:

Vectorized optimization only:
```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movups	(%rbx,%rdx,4), %xmm0
	movups	16(%rbx,%rdx,4), %xmm1
	movups	(%r15,%rdx,4), %xmm2
	addps	%xmm0, %xmm2
	movups	16(%r15,%rdx,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, (%r14,%rdx,4)
	movups	%xmm0, 16(%r14,%rdx,4)
	movups	32(%rbx,%rdx,4), %xmm0
	movups	48(%rbx,%rdx,4), %xmm1
	movups	32(%r15,%rdx,4), %xmm2
	addps	%xmm0, %xmm2
	movups	48(%r15,%rdx,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, 32(%r14,%rdx,4)
	movups	%xmm0, 48(%r14,%rdx,4)
	addq	$16, %rdx
	cmpq	$1024, %rdx             # imm = 0x400
	jne	.LBB0_3
	jmp	.LBB0_4
```

Vectorized optimization and AVX2:
```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovaps	(%rbx,%rcx,4), %ymm0
	vmovaps	32(%rbx,%rcx,4), %ymm1
	vmovaps	64(%rbx,%rcx,4), %ymm2
	vmovaps	96(%rbx,%rcx,4), %ymm3
	vaddps	(%r15,%rcx,4), %ymm0, %ymm0
	vaddps	32(%r15,%rcx,4), %ymm1, %ymm1
	vaddps	64(%r15,%rcx,4), %ymm2, %ymm2
	vaddps	96(%r15,%rcx,4), %ymm3, %ymm3
	vmovaps	%ymm0, (%r14,%rcx,4)
	vmovaps	%ymm1, 32(%r14,%rcx,4)
	vmovaps	%ymm2, 64(%r14,%rcx,4)
	vmovaps	%ymm3, 96(%r14,%rcx,4)
	addq	$32, %rcx
	cmpq	$1024, %rcx             # imm = 0x400
	jne	.LBB0_3
```

Since the offset between the instructions of both methods is 16 and 32, the default vector registers and AVX2 vector registers should be 16*8=128 and 32*8=256 bits, respectively, to process the instruction within a single cycle.

### Parallelism for branch operation
> Q2-3: Provide a theory for why the compiler is generating dramatically different assembly.

Since there's a branch condition with only ```if``` block to assign a value when the condition meets, the compiler cannot identify the similarity between each array element. Some elements should perform ```if``` block operation while others don't, making the dataflow somewhat "irregular." Thus, the compiler interprets the original C++ source code as a regular sequential operation without parallelism. The modified version uses a complete ```if else``` statement to inform the compiler that the code should perform only one assignment operation based on the acomparison result. This way, it is easier for the compiler to generate the same instruction for each element and do further parallelism optimization.

Original C++ & assembly code:
```cpp
    c[j] = a[j];
    if (b[j] > a[j])
        c[j] = b[j];
```

```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	(%r15,%rcx,4), %edx
	movl	%edx, (%rbx,%rcx,4)
	movss	(%r14,%rcx,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	movd	%edx, %xmm1
	ucomiss	%xmm1, %xmm0
	jbe	.LBB0_5
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	movss	%xmm0, (%rbx,%rcx,4)
.LBB0_5:                                #   in Loop: Header=BB0_3 Depth=2
	movl	4(%r15,%rcx,4), %edx
	movl	%edx, 4(%rbx,%rcx,4)
	movss	4(%r14,%rcx,4), %xmm0   # xmm0 = mem[0],zero,zero,zero
	movd	%edx, %xmm1
	ucomiss	%xmm1, %xmm0
	jbe	.LBB0_7
# %bb.6:                                #   in Loop: Header=BB0_3 Depth=2
	movss	%xmm0, 4(%rbx,%rcx,4)
	jmp	.LBB0_7
```

Modified C++ & assembly code:
```cpp
    if (b[j] > a[j]) c[j] = b[j];
    else c[j] = a[j];
```

```asm
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movaps	(%r15,%rcx,4), %xmm0
	movaps	16(%r15,%rcx,4), %xmm1
	maxps	(%rbx,%rcx,4), %xmm0
	maxps	16(%rbx,%rcx,4), %xmm1
	movups	%xmm0, (%r14,%rcx,4)
	movups	%xmm1, 16(%r14,%rcx,4)
	movaps	32(%r15,%rcx,4), %xmm0
	movaps	48(%r15,%rcx,4), %xmm1
	maxps	32(%rbx,%rcx,4), %xmm0
	maxps	48(%rbx,%rcx,4), %xmm1
	movups	%xmm0, 32(%r14,%rcx,4)
	movups	%xmm1, 48(%r14,%rcx,4)
	addq	$16, %rcx
	cmpq	$1024, %rcx             # imm = 0x400
	jne	.LBB0_3
# %bb.4:                                #   in Loop: Header=BB0_2 Depth=1
	addl	$1, %eax
	cmpl	$20000000, %eax         # imm = 0x1312D00
	jne	.LBB0_2
```

As shown above, the original assembly code compares every scalar independently using ```UCOMISS``` (Unordered Compare Scalar Single-Precision Floating-Point Values and Set EFLAGS) instruction and then performs required operations  if the condition has been met. Modified assembly code uses ```MAXAPS```, ```MOVUPS```, and ```MOVAPS``` to process data parallelly.

### Reference
1. [Wikipeida: Advanced Vector Extensions](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
2. [Intel® Advanced Vector Extensions Programming Reference](https://www.intel.com/content/dam/develop/external/us/en/documents/36945)
3. [Data Alignment to Assist Vectorization](https://www.intel.com/content/www/us/en/developer/articles/technical/data-alignment-to-assist-vectorization.html)
4. [x86 and amd64 instruction reference](https://www.felixcloutier.com/x86/index.html)
5. [What does gcc's ffast-math actually do?](https://stackoverflow.com/questions/7420665/what-does-gccs-ffast-math-actually-do)