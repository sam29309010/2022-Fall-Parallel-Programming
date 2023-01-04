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
    long long int local_hit = 0, global_hit = 0;
    double x, y, dist;

    // TODO: init MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    unsigned int seed = world_rank * time(NULL);
    for (long long int i = 0; i < tosses; i += world_size){
        x = (double) rand_r(&seed) / RAND_MAX;
        y = (double) rand_r(&seed) / RAND_MAX;
        dist = x * x + y * y;
        if (dist <= 1)
            local_hit++;
    }

    if (world_rank > 0)
    {
        // TODO: handle workers
        MPI_Send(&local_hit, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
        global_hit += local_hit;
        for (int i = 1; i < world_size; i++)
        {
            /* code */
            MPI_Recv(&local_hit, 1, MPI_LONG_LONG_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_hit += local_hit;
        }
        
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        pi_result = 4 * global_hit / ((double) tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
