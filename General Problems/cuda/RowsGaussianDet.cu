#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

__global__ static void gaussianEliminationRows(double *all_matrices, int matrix_order, int matrices_count);
__device__ static void find_matrix_non_zero_row(double *matrix, int row_index, int column_index, int matrix_order, int *row);
__device__ static void swap_matrix_rows(double *matrix, int first_row, int second_row, int thread_index, int matrix_order);

int main(int argc, char const *argv[])
{
    // usage
    if (argc == 1)
    {
        printf("Program Usage:\n\t ./ex2 (matrix*.txt)+\n");
        return 1;
    }

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // time vars
    double exec_start = seconds();
    for (int matrix_file_i = 1; matrix_file_i < argc; matrix_file_i++)
    {
        // for every file
        FILE *matrix_file = fopen(argv[matrix_file_i], "r");
        if (matrix_file != NULL)
        {
            int matrix_count, matrix_order;
            fread(&matrix_count, sizeof(int), 1, matrix_file);
            fread(&matrix_order, sizeof(int), 1, matrix_file);
            int all_matrices_size = matrix_count * matrix_order * matrix_order * sizeof(double);
            double *all_matrices = (double *)malloc(all_matrices_size);

            // read all matrices
            for (int matrix_i = 0; matrix_i < matrix_count; matrix_i++)
            {
                double *current_matrix = &all_matrices[matrix_i * matrix_order * matrix_order];
                for (int i = 0; i < matrix_order * matrix_order; i++)
                    fread(&current_matrix[i], sizeof(double), 1, matrix_file);
            }

            fclose(matrix_file);
            double *gpu_matrices;

            // allocate memory for the matrices on the gpu and memcopy
            CHECK(cudaMalloc((void **)&gpu_matrices, all_matrices_size));
            CHECK(cudaMemcpy(gpu_matrices, all_matrices, all_matrices_size, cudaMemcpyHostToDevice));

            dim3 grid(matrix_count, 1, 1);
            dim3 block(matrix_order, 1, 1);
            gaussianEliminationRows<<<grid, block>>>(gpu_matrices, matrix_order, matrix_count);

            // error check
            CHECK(cudaGetLastError());

            // transfering the matrices back
            CHECK(cudaMemcpy(all_matrices, gpu_matrices, all_matrices_size, cudaMemcpyDeviceToHost));

            // determinant calculation for each matrix
            for (int i = 0; i < matrix_count; i++)
            {
                double determinant = 1.0;
                int current_matrix_offset = i * matrix_order * matrix_order;
                for (int j = 0; j < matrix_order; j++)
                    determinant *= all_matrices[matrix_order * j + j + current_matrix_offset];
                fprintf(stdout, "\nMatrix %d: %E", (i + 1), determinant);
            }

            // memory frees and gpu device reset
            free(all_matrices);
            CHECK(cudaFree(gpu_matrices));
            CHECK(cudaDeviceReset());
        }
        else
            printf("The provided file: %s\n couldn't be opened!\n", argv[matrix_file_i]);
    }

    // execution time
    double exec_time = seconds() - exec_start;
    fprintf(stdout, "\n\nTime Elapsed: %f s\n", exec_time);
    return 0;
}

__global__ static void gaussianEliminationRows(double *all_matrices, int matrix_order, int matrices_count)
{
    int matrix_i = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    int thread_ID = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    if (matrix_i >= matrices_count)
        return;
    double *current_matrix = &all_matrices[matrix_i * matrix_order * matrix_order];

    int current_row_pivot = 0;
    int current_column_pivot = 0;
    while (current_row_pivot < matrix_order && current_column_pivot < matrix_order)
    {
        if (current_matrix[current_row_pivot * matrix_order + current_column_pivot] == 0)
        {
            int row = -1;
            find_matrix_non_zero_row(current_matrix, current_row_pivot, current_column_pivot, matrix_order, &row);

            __syncthreads();

            // if there is no non zero row just set the value to 0 so that the determinant will also be 0 and return
            if (row == -1)
                return;

            // swap rows
            if (row != current_row_pivot)
                swap_matrix_rows(current_matrix, current_row_pivot, row, thread_ID, matrix_order);

            __syncthreads();
        }

        // Calculate the gaussian elimination factor first
        for (int next_row = current_row_pivot + 1; next_row < matrix_order; next_row++)
        {
            double factor;
            if (thread_ID < matrix_order && thread_ID >= current_column_pivot)
            {
                // matrix[i][i]
                int pivot_index = current_row_pivot * matrix_order + current_column_pivot;
                // matrix[k][i]
                int next_row_index = next_row * matrix_order + current_column_pivot;
                // matrix[k][i] / matrix[i][i]
                factor = current_matrix[next_row_index] / current_matrix[pivot_index];
            }

            __syncthreads();

            // apply the transformation for the remaining elements in the same row on the next columns

            if (thread_ID < matrix_order && thread_ID >= current_column_pivot)
            {
                // matrix[k][j]
                int thread_row_index = next_row * matrix_order + thread_ID;
                // matrix[i][j]
                int factor_element_index = current_row_pivot * matrix_order + thread_ID;
                // matrix[k][j] = matrix[k][j] - (matrix[k][i] / matrix[i][i]) * matrix[i][j]
                current_matrix[thread_row_index] -= factor * current_matrix[factor_element_index];
            }
        }
        // wait and move to the next pivot
        __syncthreads();

        current_row_pivot++;
        current_column_pivot++;
    }
}

// Function for a thread to find a non zero row
__device__ static void find_matrix_non_zero_row(double *matrix, int row_index, int column_index, int matrix_order, int *row)
{
    *row = -1;
    for (int i = row_index; i < matrix_order; i++)
    {
        if (matrix[i * matrix_order + column_index] != 0.0)
        {
            *row = i;
            break;
        }
    }
}
__device__ static void swap_matrix_rows(double *matrix, int first_row, int second_row, int thread_index, int matrix_order)
{
    int first_row_index = first_row * matrix_order + thread_index;
    int second_row_index = second_row * matrix_order + thread_index;
    double temp = matrix[first_row_index];
    matrix[first_row_index] = matrix[second_row_index];
    matrix[second_row_index] = temp;
}