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
    long long int hit = 0;
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
            hit++;
    }

    // TODO: use MPI_Gather
    long long int *recv_hit_buf = (long long int *) malloc(world_size * sizeof(long long int)); // used only in host node
    MPI_Gather(&hit, 1, MPI_LONG_LONG_INT, recv_hit_buf, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    if (world_rank == 0)
    {
        // TODO: PI result
        for (int i = 1; i < world_size; i++)
        {
            hit += recv_hit_buf[i];
        }
        pi_result = 4 * hit / ((double) tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    free(recv_hit_buf);
    return 0;
}
