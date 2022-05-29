#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "my_utils.h"

#define WORKTODO 1
#define NOMOREWORK 0

int main(int argc, char *argv[])
{
    int rank, size;
    unsigned int whatToDo; /* command */
    unsigned int current_worker = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int results_size = 100;
    int current_result = 0;
    if (argc == 1)
    {
        MPI_Finalize();
        if (rank == 0)
            printf("Usage:\n\tThe program should be called with at least one matrix file as an argument!\n");
        return EXIT_SUCCESS;
    }
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT to sincronize the start! */
    double start_time = MPI_Wtime();
    if (rank == 0)
    {
        double *results = calloc(100, sizeof(double));
        int matrix_count;
        for (int matrix_file_i = 1; matrix_file_i < argc; matrix_file_i++)
        {
            FILE *matrix_file = fopen(argv[matrix_file_i], "r");
            if (matrix_file == NULL)
            {
                printf("File %s couldn't be opened!\n", argv[matrix_file_i]);
                continue;
            }
            int matrix_order;
            fread(&matrix_count, sizeof(int), 1, matrix_file);
            fread(&matrix_order, sizeof(int), 1, matrix_file);
            printf("%d %dx%d matrixes\n", matrix_count, matrix_order, matrix_order);
            double *data = malloc(matrix_order * matrix_order * sizeof(double));
            for (int matrix_i = 0; matrix_i < matrix_count; matrix_i++)
            {
                for (int i = 0; i < matrix_order; i++)
                {
                    for (int j = 0; j < matrix_order; j++)
                        fread(&data[i * matrix_order + j], sizeof(double), 1, matrix_file);
                }
                whatToDo = WORKTODO;
                MPI_Send(&whatToDo, 1, MPI_UNSIGNED, current_worker, 0, MPI_COMM_WORLD);
                MPI_Send(&matrix_order, 1, MPI_INT, current_worker, 0, MPI_COMM_WORLD);
                MPI_Send(data, matrix_order * matrix_order, MPI_DOUBLE, current_worker, 0, MPI_COMM_WORLD);
                if (current_worker == size - 1)
                {
                    current_worker = 1;
                    for (int worker_i = 1; worker_i < size; worker_i++)
                    {
                        if (current_result == results_size)
                        {
                            results_size += 100;
                            results = realloc(results, results_size * sizeof(double));
                        }
                        MPI_Recv(&results[current_result], 1, MPI_DOUBLE, worker_i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        printf("Result %d is %E\n", current_result, results[current_result]);
                        fflush(stdout);
                        current_result += 1;
                    }
                }
                else
                    current_worker++;
            }
            free(data);
            fclose(matrix_file);
        }
        if (current_worker > 1)
        {
            for (int worker_i = 1; worker_i < current_worker; worker_i++)
            {
                if (current_result == results_size)
                {
                    results_size += current_worker;
                    results = realloc(results, results_size * sizeof(double));
                }
                MPI_Recv(&results[current_result], 1, MPI_DOUBLE, worker_i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Result %d is %E\n", current_result, results[current_result]);
                fflush(stdout);
                current_result += 1;
            }
        }
        whatToDo = NOMOREWORK;
        for (int worker_id = 1; worker_id < size; worker_id++)
            MPI_Send(&whatToDo, 1, MPI_UNSIGNED, worker_id, 0, MPI_COMM_WORLD);
        free(results);
        printf("Root process Finished!\n");
    }
    else
    {
        unsigned int recv_status;
        while (1)
        {
            MPI_Recv(&recv_status, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_status == NOMOREWORK)
                break;
            int matrix_order;
            MPI_Recv(&matrix_order, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double *data = malloc(matrix_order * matrix_order * sizeof(double));
            MPI_Recv(data, matrix_order * matrix_order, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double **matrix = malloc(matrix_order * sizeof(double *));
            for (int i = 0; i < matrix_order; i++)
            {
                matrix[i] = malloc(matrix_order * sizeof(double));
                for (int j = 0; j < matrix_order; j++)
                    matrix[i][j] = data[i * matrix_order + j];
            }
            double res = calculateSquaredMatrixDeterminant(matrix, matrix_order);
            MPI_Send(&res, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            free(data);
            for (int i = 0; i < matrix_order; i++)
                free(matrix[i]);
            free(matrix);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    double end_time = MPI_Wtime();
    MPI_Finalize();
    if (rank == 0)
    { /* use time on master node */
        printf("Execution time: %f s\n", end_time - start_time);
    }
    return EXIT_SUCCESS;
}
