#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "fifo.h"

#define nConsumers 100
/** \brief producer threads return status array */
// int statusProd[nConsumers];

static int status = 1;

typedef struct producer_args_t
{
    FILE **matrix_files;
    int total_files;
};

typedef struct consumer_res_t
{
    double det;
    int matrix_id;
};

typedef struct consumer_args_t
{
    struct consumer_res_t *results;
    int size;
};

void printSquaredMatrix(double **matrix, int size)
{
    char number_buffer[20];
    char res[100000];
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            sprintf(number_buffer, "%e ", matrix[i][j]);
            strcat(res, number_buffer);
        }
        strcat(res, "\n\n");
    }
    printf("%s\n", res);
    fflush(stdout);
}

int findAndSwapCols(double **a, int size, int col_in)
{
    double *first_col = calloc(size, sizeof(double));
    double *second_col = calloc(size, sizeof(double));
    int second_col_index = size;
    for (int i = 0; i < size; i++)
    {
        first_col[i] = a[i][col_in];
        if (i > col_in && a[col_in][i] != 0)
            second_col_index = i;
    }
    if (second_col_index == size)
        return 0;
    for (int i = 0; i < size; i++)
        second_col[i] = a[i][second_col_index];
    for (int i = 0; i < size; i++)
    {
        a[i][col_in] = second_col[i];
        a[i][second_col_index] = first_col[i];
    }
    free(first_col);
    free(second_col);
    return -1;
}

int gaussianElimination(double **a, int matrix_order)
{ // returns the determinant sign (+det / -det)
    int res_sign = 1;
    for (int i = 0; i < matrix_order - 1; i++)
    {
        if (a[i][i] == 0)
        {
            int swap_res = findAndSwapCols(a, matrix_order, i);
            if (swap_res == 0)
                return 0;
            res_sign *= swap_res;
        }
        for (int k = i + 1; k < matrix_order; k++)
        {
            double ratio = (a[k][i] / a[i][i]);
            for (int j = i; j < matrix_order; j++)
                a[k][j] -= ratio * a[i][j];
        }
    }
    return res_sign;
}

void *calculateDeterminant(void *result)
{
    int current_matrix_i = 0;
    while (1)
    {

        struct consumer_args_t *res = (struct consumer_args_t *)result;
        // printf("%d\n",status);
        struct matrix_t *matrix = getVal();
        if (matrix == NULL)
        {
            res->size = current_matrix_i;
            pthread_exit(EXIT_SUCCESS);
        }
        int det_sign = gaussianElimination(matrix->matrix, matrix->size);
        if (det_sign == 0)
            return 0;
        double det = det_sign;
        for (int i = 0; i < matrix->size; i++)
        {
            det *= matrix->matrix[i][i];
        }
        struct consumer_res_t current_res;
        current_res.det = det;
        current_res.matrix_id = matrix->id;
        if (current_matrix_i >= res->size)
        {
            res->results = realloc(res->results, (res->size + 10) * sizeof(struct consumer_res_t));
            res->size += 10;
        }
        res->results[current_matrix_i] = current_res;
        for (int i = 0; i < matrix->size; i++)
            free(matrix->matrix[i]);
        free(matrix->matrix);
        free(matrix);
        current_matrix_i++;
        usleep((unsigned int)floor(40.0 * random() / RAND_MAX + 1.5)); /* do something else */
    }
}
void *produceMatrix(void *matrix_file_arg)
{
    struct producer_args_t *args = (struct producer_args_t *)matrix_file_arg;
    for (int matrix_file_i = 0; matrix_file_i < args->total_files; matrix_file_i++)
    {
        FILE *matrix_file = args->matrix_files[matrix_file_i];
        if (matrix_file == NULL)
        {
            printf("File couldn't be opened!");
            continue;
        }
        int matrix_count, matrix_order;
        fread(&matrix_count, sizeof(int), 1, matrix_file);
        fread(&matrix_order, sizeof(int), 1, matrix_file);
        for (int matrix_i = 0; matrix_i < matrix_count; matrix_i++)
        {
            double **matrix = calloc(matrix_order, sizeof(double *));
            for (int i = 0; i < matrix_order; i++)
            {
                matrix[i] = calloc(matrix_order, sizeof(double));
                for (int j = 0; j < matrix_order; j++)
                    fread(&matrix[i][j], sizeof(double), 1, matrix_file);
            }
            struct matrix_t *matrix_struct = calloc(1, sizeof(struct matrix_t));
            matrix_struct->id = (matrix_file_i + 1) * (matrix_i + 1);
            matrix_struct->matrix = matrix;
            matrix_struct->size = matrix_order;
            putVal(matrix_struct);
            usleep((unsigned int)floor(40.0 * random() / RAND_MAX + 1.5)); /* do something else */
        }
        fclose(args->matrix_files[matrix_file_i]);
    }
    status = EXIT_SUCCESS;
    setDone();
    pthread_exit(&status);
}

int main(int argc, char *argv[])
{
    printf("Running ex2!\n\n");

    if (argc == 1)
    {
        printf("Program Usage:\n\t ./ex2 (matrix*.txt)+\n");
        return 1;
    }
    struct timespec start, finish;

    clock_gettime(CLOCK_MONOTONIC, &start);
    pthread_t producer;
    FILE **files = calloc(argc - 1, sizeof(FILE *));
    for (int matrix_i = 1; matrix_i < argc; matrix_i++)
    {
        files[matrix_i - 1] = fopen(argv[matrix_i], "r");
    }
    struct producer_args_t *args = calloc(1, sizeof(struct producer_args_t));
    args->matrix_files = files;
    args->total_files = argc - 1;
    if (pthread_create(&producer, NULL, produceMatrix, (void *)args) != 0)
    {
        perror("error on creating thread producer");
        exit(EXIT_FAILURE);
    }
    pthread_t matrixSolvers[nConsumers];
    struct consumer_args_t *detResults = calloc(nConsumers, sizeof(struct consumer_args_t));
    for (int i = 0; i < nConsumers; i++)
    {
        detResults[i].size = 10;
        detResults[i].results = malloc(10 * sizeof(struct consumer_res_t));
        pthread_create(&matrixSolvers[i], NULL, calculateDeterminant, (void *)&detResults[i]);
    }
    pthread_join(producer, NULL);
    for (int i = 0; i < nConsumers; i++)
    {
        pthread_join(matrixSolvers[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    
    for (int i = 0; i < nConsumers; i++)
    {
        //    for (int result_i = 0; result_i < detResults->size; result_i++)
        //    {
        //        printf("Processing Matrix %d:\n The determinant is: %e\n", detResults[i].results[result_i].matrix_id, detResults[i].results[result_i].det);
        //    }
        free(detResults[i].results);
    }
    free(detResults);
    free(args);
    free(files);
    double time_taken = (finish.tv_sec - start.tv_sec);
    time_taken += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Elapsed time = %lf s\n", time_taken);
    return 0;
}