#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---
    // hit counter used for send / recvieve
    long long int send_hit = 0, recv_hit = 0;
    double x, y, dist;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    unsigned int seed = world_rank * time(NULL);
    for (long long int i = 0; i < tosses; i += world_size){
        x = (double) rand_r(&seed) / RAND_MAX;
        y = (double) rand_r(&seed) / RAND_MAX;
        dist = x * x + y * y;
        if (dist <= 1)
            send_hit++;
    }

    // TODO: binary tree redunction
    int offset = 1;
    while (offset < world_size)
    {
        if (world_rank % (offset <<1) == offset){
            MPI_Send(&send_hit, 1, MPI_LONG_LONG_INT, world_rank - offset, 0, MPI_COMM_WORLD);
            break;
        }
        else if (world_rank % (offset <<1) == 0){
            MPI_Recv(&recv_hit, 1, MPI_LONG_LONG_INT, world_rank + offset, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            send_hit += recv_hit;
        }
        offset <<= 1;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * send_hit / ((double) tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}