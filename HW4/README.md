Parallel Programming Lab4 Report: MPI Programming
===

###### tags: `PP`

* Lab4 Report of CSIC30148 2022 Fall Parallel Programming @ NYCU
* Editor: 310551145 Cheng-Che Lu

### TL;DR
*OpenMPI* provides a message-passing interface that enables communication across processes. The communication overhead, bandwidth, and mechanism (e.g., blocking or non-blocking) should be considered when using *OpenMPI*.

### Basic Configurations Settings
> Q1.1 How do you control the number of MPI processes on each node?

The argument ***--hostfile hosts*** could control the number of MPI processes on each node. *hosts* is a file that contains the information of available nodes and corresponding maximum processes as below.


```
pp2 slots=1
pp3 slots=1
pp4 slots=1
pp5 slots=1
```

> Q1.2 Which functions do you use for retrieving the rank of an MPI process and the total number of processes?

Function ***MPI_Comm_size*** could retrieve the rank of an MPI process, while function ***MPI_Comm_rank*** could retrieve the total number of processes.

```cpp
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
```

### Basic MPI Operation and Performance Comparison
#### *MPI_Send* & *MPI_Recv*
>Q2.1 Why MPI_Send and MPI_Recv are called “blocking” communication?

*MPI_Send* and *MPI_Recv* are called "blocking" communication **since they do not return (i.e., they block) until the communication is finished.** A *MPI_Send* call must wait until one *MPI_Recv* call receives its data, while a *MPI_Recv* call must wait until one *MPI_Send* call transmits its data.

>Q2.2 Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

| # Process | MPI Running Time |
| --------- | ---------------- |
| 2         |  6.50 sec        |
| 4         |  3.36 sec        |
| 8         |  1.68 sec        |
| 12        |  1.15 sec        |
| 16        |  0.93 sec        |

![](https://i.imgur.com/42mo2Bw.png)



<!--

mpirun -np 4 --hostfile hosts.txt pi_block_linear 1000000000
mpirun -np 4 --hostfile hosts.txt /HW4/ref/pi_block_linear 1000000000

mpirun -np 1 --hostfile hosts_all.txt pi_block_linear 1000000000
mpirun -np 2 --hostfile hosts_all.txt pi_block_linear 1000000000
mpirun -np 4 --hostfile hosts_all.txt pi_block_linear 1000000000
mpirun -np 8 --hostfile hosts_all.txt pi_block_linear 1000000000
mpirun -np 12 --hostfile hosts_all.txt pi_block_linear 1000000000
mpirun -np 16 --hostfile hosts_all.txt pi_block_linear 1000000000
-->

#### Binary Tree Reduction Communication Algorithm
>Q3.1 Measure the performance (execution time) of the code for 2, 4, 8, 16 MPI processes and plot it.

| # Process | MPI Running Time |
| --------- | ---------------- |
| 2         |  6.43 sec         |
| 4         |  3.32 sec         |
| 8         |  1.71 sec         |
| 16        |  0.88 sec         |

![](https://i.imgur.com/mLHQAqT.png)


<!--
mpirun -np 4 --hostfile hosts.txt pi_block_tree 1000000000
mpirun -np 4 --hostfile hosts.txt /HW4/ref/pi_block_tree 1000000000

mpirun -np 2 --hostfile hosts_all.txt pi_block_tree 1000000000
mpirun -np 4 --hostfile hosts_all.txt pi_block_tree 1000000000
mpirun -np 8 --hostfile hosts_all.txt pi_block_tree 1000000000
mpirun -np 16 --hostfile hosts_all.txt pi_block_tree 1000000000
-->



>Q3.2 How does the performance of binary tree reduction compare to the performance of linear reduction?

The performance of the binary tree reduction program is **slightly faster** than the linear reduction one. The program spends most time generating a 2-d random number and its norm, so the effect of different communication strategies is little in this case.

>Q3.3 Increasing the number of processes, which approach (linear/tree) is going to perform better? Why? Think about the number of messages and their costs.

The binary tree reduction program would be better if increasing the number of processes. On the one hand, a binary tree reduction program requires maintaining the partial sum result in tree structure across the process, which is a slight overhead for the program. On the other hand, the binary tree reduction program **reduces the communication overhead of host node** by pre-accumulating the summation in different nodes, which may improve the performance, especially when many processes are working together.

#### Non-blocking communication
>Q4.1 Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

| # Process | MPI Running Time |
| --------- | ---------------- |
| 2         |  6.56 sec         |
| 4         |  3.30 sec         |
| 8         |  1.84 sec         |
| 12        |  1.12 sec         |
| 16        |  0.89 sec         |

![](https://i.imgur.com/E2Rcsb5.png)


<!--
mpirun -np 4 --hostfile hosts.txt pi_nonblock_linear 1000000000
mpirun -np 4 --hostfile hosts.txt /HW4/ref/pi_nonblock_linear 1000000000

mpirun -np 2 --hostfile hosts_all.txt pi_nonblock_linear 1000000000
mpirun -np 4 --hostfile hosts_all.txt pi_nonblock_linear 1000000000
mpirun -np 8 --hostfile hosts_all.txt pi_nonblock_linear 1000000000
mpirun -np 12 --hostfile hosts_all.txt pi_nonblock_linear 1000000000
mpirun -np 16 --hostfile hosts_all.txt pi_nonblock_linear 1000000000
-->



>Q4.2 What are the MPI functions for non-blocking communication?

The commonly-used non-blocking communication functions are as follows.

```cpp
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
              MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
              int tag, MPI_Comm comm, MPI_Request * request);
int MPI_Wait(MPI_Request *request, MPI_Status *status);
int MPI_Waitany(int count, MPI_Request array_of_requests[], int *indx, MPI_Status * status);
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
int MPI_Testany(int count, MPI_Request array_of_requests[], int *indx,
               int *flag, MPI_Status *status);
```

Also, MPI functions with prefix **I** are normally a non-blocking one.

>Q4.3 How the performance of non-blocking communication compares to the performance of blocking communication?

The performance of non-blocking communication is **basically the same** as that of blocking one. As *Q3.2* has mentioned, the program spends most time generating a 2-d random number and its norm and transmits only an integer from/to another node. Hence, the effect of different communication strategies is little in this case.

#### Collective communication: Gather & Reduce
>Q5 Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

***MPIGather***
| # Process | MPI Running Time |
| --------- | ---------------- |
| 2         |  6.46 sec         |
| 4         |  3.30 sec         |
| 8         |  1.91 sec         |
| 12        |  1.35 sec         |
| 16        |  0.85 sec         |

![](https://i.imgur.com/zTwokmA.png)


<!--
mpirun -np 4 --hostfile hosts.txt pi_gather 1000000000
mpirun -np 4 --hostfile hosts.txt /HW4/ref/pi_gather 1000000000

mpirun -np 2 --hostfile hosts_all.txt pi_gather 1000000000
mpirun -np 4 --hostfile hosts_all.txt pi_gather 1000000000
mpirun -np 8 --hostfile hosts_all.txt pi_gather 1000000000
mpirun -np 12 --hostfile hosts_all.txt pi_gather 1000000000
mpirun -np 16 --hostfile hosts_all.txt pi_gather 1000000000
-->


>Q6 Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

***Reduce***
| # Process | MPI Running Time |
| --------- | ---------------- |
| 2         |  6.58 sec         |
| 4         |  3.29 sec         |
| 8         |  1.68 sec         |
| 12        |  1.13 sec         |
| 16        |  0.86 sec         |

![](https://i.imgur.com/gzwsE8q.png)


<!--
mpirun -np 4 --hostfile hosts.txt pi_reduce 1000000000
mpirun -np 4 --hostfile hosts.txt /HW4/ref/pi_reduce 1000000000

mpirun -np 2 --hostfile hosts_all.txt pi_reduce 1000000000
mpirun -np 4 --hostfile hosts_all.txt pi_reduce 1000000000
mpirun -np 8 --hostfile hosts_all.txt pi_reduce 1000000000
mpirun -np 12 --hostfile hosts_all.txt pi_reduce 1000000000
mpirun -np 16 --hostfile hosts_all.txt pi_reduce 1000000000
-->

### Matrix Multiplication with MPI
>Q7 Describe what approach(es) were used in your MPI matrix multiplication for each data set.

To compute matrix multiplication ````A @ B = C```` with multiple processes using MPI, matrix A is divided and scattered along the row axis (```MPI_Scatter```) while the whole matrix B is simply broadcast (```MPI_Bcast```) to each process. Each process computes several matrix C rows and sends them back to the host (```MPI_Gather```). In other words, the **splitting** approach is used to spread the data so that processors can utilize parallelism to improve performance.

### Reference
1. [Difference between *rand()* and *randr()*](https://sam66.medium.com/rand-%E8%88%87-rand-r%E7%9A%84%E5%B7%AE%E7%95%B0-70c4bfc201f6)