#include <mpi.h>
#include <stdio.h>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0){
        int r = scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
    }
    getchar(); // next line

	MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int mat_a_size = (*n_ptr) * (*m_ptr);
    const int mat_b_size = (*m_ptr) * (*l_ptr);

    // may use char data type
    *a_mat_ptr = (int *) calloc(mat_a_size, sizeof(int));
    *b_mat_ptr = (int *) calloc(mat_b_size, sizeof(int));

    int *a_mat = (int *) *a_mat_ptr;
    int *b_mat = (int *) *b_mat_ptr;

    if (world_rank == 0)
    {
        // row major
        for (int i = 0; i < mat_a_size; i++)
        {
            char ch = getchar();
            while ('0' <= ch && ch <= '9')
            {
                a_mat[i] = a_mat[i] * 10 + ch - '0';
                ch = getchar();
            }
            if (i % (*m_ptr) == (*m_ptr) - 1)
                getchar();
        }

        // columns major
        for (int i = 0; i < *m_ptr; i++)
        {
            for (int j = 0; j < *l_ptr; j++)
            {
                int index = j * (*m_ptr) + i;
                char ch = getchar();
                while ('0' <= ch && ch <= '9')
                {
                    b_mat[index] = b_mat[index] * 10 + ch - '0';
                    ch = getchar();
                }
            }
            getchar();
        }
    }

    const int average_count = (*n_ptr) / world_size * (*m_ptr);
    const int remains_count = (*n_ptr) % world_size * (*m_ptr);

    int *sendbuf = *a_mat_ptr + remains_count;
    int *recvbuf = *a_mat_ptr + ((world_rank == 0) ? remains_count : 0);
    MPI_Scatter(sendbuf, average_count, MPI_INT, recvbuf, average_count, MPI_INT, 0, MPI_COMM_WORLD);
    // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //             void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    //             MPI_Comm comm)
	MPI_Bcast(*b_mat_ptr, mat_b_size, MPI_INT, 0, MPI_COMM_WORLD);

    // printf("Process %d construct_matrices: %lf Seconds\n",world_rank ,MPI_Wtime());
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int mat_c_size = n * l;
    int *c_mat = (int *) calloc(mat_c_size, sizeof(int));

    const int average_row = n / world_size;
    const int remains_row = n % world_size;
    const int average_count = average_row * l;
    const int remains_count = remains_row * l;

    const int end_row = (world_rank == 0) ? average_row + remains_row : average_row;

    // int a_index, b_index, c_index;
    // int sum;
    // for (int row = 0; row < end_row; row++)
    // {
    //     a_index = row * m;
    //     c_index = row * l;
    //     for (int col = 0; col < l; col++)
    //     {
    //         b_index = col * m;
    //         sum = 0;
    //         for (int k = 0; k < m; k++)
    //         {
    //             // sum += A[r,k] * B[k,c]
    //             sum += a_mat[a_index + k] * b_mat[b_index + k];
    //         }
    //         c_mat[c_index + col] = sum;
    //     }
    // }
    int a_index = 0, a_index_p_m = m,  b_index = 0, c_index = 0;
    int sum = 0;
    for (int row = 0; row < end_row; row++, a_index+= m, a_index_p_m+= m, b_index=0)
    {
        for (int col = 0; col < l; col++, sum = 0)
        {
            for (int k = a_index; k < a_index_p_m; )
            {
                // sum += A[r,k] * B[k,c]
                sum += a_mat[k++] * b_mat[b_index++];
            }
            c_mat[c_index++] = sum;
        }
    }

    // MPI_Reduce(c_mat, ans_mat, mat_c_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int *sendbuf = (world_rank == 0) ? c_mat + remains_count : c_mat;
    int *recvbuf = c_mat + remains_count;
    MPI_Gather(sendbuf, average_count, MPI_INT, recvbuf, average_count, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //            void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)

    c_index = 0;
	if(world_rank == 0){
		for(int i = 0 ; i < n ; i++){
            for (int j = 0; j < l; j++)
                printf("%d ", c_mat[c_index++]);
            printf("\n");
		}
	}

    // free c_mat and ans
    // printf("Process %d matrix_multiply: %lf Seconds\n", world_rank ,MPI_Wtime());
}

// void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){
//     int world_rank, world_size;
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

//     if (world_rank == 0){
//         int r = scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
//     }
//     getchar(); // next line

// 	MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     const int mat_a_size = (*n_ptr) * (*m_ptr);
//     const int mat_b_size = (*m_ptr) * (*l_ptr);

//     *a_mat_ptr = (int *) calloc(mat_a_size, sizeof(int));
//     *b_mat_ptr = (int *) calloc(mat_b_size, sizeof(int));

//     int *a_mat = (int *) *a_mat_ptr;
//     int *b_mat = (int *) *b_mat_ptr;

//     if (world_rank == 0)
//     {
//         // row major
//         for (int i = 0; i < mat_a_size; i++)
//         {
//             char ch = getchar();
//             while ('0' <= ch && ch <= '9')
//             {
//                 a_mat[i] = a_mat[i] * 10 + ch - '0';
//                 ch = getchar();
//             }
//             if (i % (*m_ptr) == (*m_ptr) - 1)
//                 getchar();
//         }

//         // columns major
//         for (int i = 0; i < *m_ptr; i++)
//         {
//             for (int j = 0; j < *l_ptr; j++)
//             {
//                 int index = j * (*m_ptr) + i;
//                 char ch = getchar();
//                 while ('0' <= ch && ch <= '9')
//                 {
//                     b_mat[index] = b_mat[index] * 10 + ch - '0';
//                     ch = getchar();
//                 }
//             }
//             getchar();
//         }
//     }

// 	MPI_Bcast(*a_mat_ptr, mat_a_size, MPI_INT, 0, MPI_COMM_WORLD);
// 	MPI_Bcast(*b_mat_ptr, mat_b_size, MPI_INT, 0, MPI_COMM_WORLD);

//     // printf("Process %d construct_matrices: %lf Seconds\n",world_rank ,MPI_Wtime());
// }

// void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){
//     int world_rank, world_size;
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

//     const int mat_c_size = n * l;
//     int *c_mat = (int *) calloc(mat_c_size, sizeof(int));
//     int *ans_mat = (int *) calloc(mat_c_size, sizeof(int));

//     int start_row = n / world_size * world_rank;
//     int end_row = (world_rank != world_size - 1) ? start_row + n / world_size : n;

//     int a_index, b_index, c_index;
//     int sum;
//     for (int row = start_row; row < end_row; row++)
//     {
//         a_index = row * m;
//         c_index = row * l;
//         for (int col = 0; col < l; col++)
//         {
//             b_index = col * m;
//             sum = 0;
//             for (int k = 0; k < m; k++)
//             {
//                 // sum += A[r,k] * B[k,c]
//                 sum += a_mat[a_index + k] * b_mat[b_index + k];
//             }
//             c_mat[c_index + col] = sum;
//         }
//     }
//     MPI_Reduce(c_mat, ans_mat, mat_c_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

// 	if(world_rank == 0){
// 		for(int i = 0 ; i < n ; i++){
//             c_index = i * l;
//             for (int j = 0; j < l; j++)
//                 printf("%d ", ans_mat[c_index++]);
//             printf("\n");
// 		}
// 	}

//     // free c_mat and ans
//     // printf("Process %d matrix_multiply: %lf Seconds\n", world_rank ,MPI_Wtime());
// }

void destruct_matrices(int *a_mat, int *b_mat){
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if(world_rank == 0){
		free(a_mat);
		free(b_mat);
	}
    // printf("Process %d destruct_matrices: %lf Seconds\n",world_rank ,MPI_Wtime());
}