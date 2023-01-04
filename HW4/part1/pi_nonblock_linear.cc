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

    // TODO: MPI init
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
        // TODO: MPI workers
        MPI_Request request;
        MPI_Isend(&local_hit, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD, &request);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request *requests = (MPI_Request *) malloc((world_size - 1) * sizeof(MPI_Request));
        long long int *local_hit_buf = (long long int *) malloc((world_size - 1) * sizeof(long long int));
        for (int i = 1; i < world_size; i++)
        {
            /* code */
            MPI_Irecv(&local_hit_buf[i-1], 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD, &requests[i-1]);
        }
        MPI_Waitall(world_size - 1, requests, MPI_STATUS_IGNORE);

        // TODO: PI result
        global_hit += local_hit;
        for (int i = 0; i < world_size - 1; i++)
        {
            global_hit += local_hit_buf[i];
        }
        pi_result = 4 * global_hit / ((double) tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
        free(requests);
        free(local_hit_buf);
    }

    // if (world_rank == 0)
    // {
    //     // TODO: PI result
    //     for (int i = 1; i < world_size; i++)
    //     {
    //         global_hit += local_hit_buf[i];
    //     }
    //     pi_result = 4 * global_hit / ((double) tosses);
    //     // --- DON'T TOUCH ---
    //     double end_time = MPI_Wtime();
    //     printf("%lf\n", pi_result);
    //     printf("MPI running time: %lf Seconds\n", end_time - start_time);
    //     // ---
    //     free(requests);
    //     free(local_hit_buf);
    // }

    MPI_Finalize();
    return 0;
}
