#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

__global__ void gaussianEliminationRows(double* all_matrixs, int matrix_order);

int main(int argc, char const *argv[])
{
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

	// Time vars
    double exec_start = seconds();
    for (int matrix_file_i = 1; matrix_file_i < argc; matrix_file_i++)
    {
    	//For every file
        FILE *matrix_file = fopen(argv[matrix_file_i], "r");
        if (matrix_file != NULL)
        {
            int matrix_count, matrix_order;
            fread(&matrix_count, sizeof(int), 1, matrix_file);
            fread(&matrix_order, sizeof(int), 1, matrix_file);
			int all_matrixes_size = matrix_count * matrix_order* matrix_order* sizeof(double);
			double* all_matrixes = (double *) malloc(all_matrixes_size);

            for (int matrix_i = 0; matrix_i < matrix_count; matrix_i++)
            {
				double *current_matrix = &all_matrixes[matrix_i * matrix_order * matrix_order];
				for (int i = 0; i < matrix_order*matrix_order; i++) 
					fread(&current_matrix[i], sizeof(double), 1, matrix_file);
            }
            fclose(matrix_file);
			double *gpu_matrixes;
			CHECK(cudaMalloc((void**) &gpu_matrixes, all_matrixes_size));
			CHECK(cudaMemcpy(&gpu_matrixes[0], &all_matrixes[0], all_matrixes_size, cudaMemcpyHostToDevice));
			
            
            dim3 grid(matrix_count, 1, 1);
			dim3 block(matrix_order, 1, 1);
            gaussianEliminationRows<<<grid, block>>>(gpu_matrixes, matrix_order);

            //CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
			CHECK(cudaMemcpy(&all_matrixes[0], &gpu_matrixes[0], all_matrixes_size, cudaMemcpyHostToDevice));
            for(int i = 0; i < matrix_count; i++) {
                double determinant = 1.0;
                int current_matrix_offset = i * matrix_order * matrix_order;
                for(int j = 0; j < matrix_order; j++) 
                    determinant *= all_matrixes[matrix_order * j + j + current_matrix_offset];
                fprintf(stdout, "\nMatrix %d: %E", (i + 1), determinant);
            }

			free(all_matrixes);
			CHECK(cudaFree(gpu_matrixes));
            CHECK(cudaDeviceReset());
            fclose(matrix_file);
        }
        else
            printf("The provided file: %s\n couldn't be opened!\n", argv[matrix_file_i]);
    }

    double exec_time = seconds() - exec_start;
        fprintf(stdout, "\n\nTime Elapsed: %.3e s\n", exec_time);
    CHECK(cudaDeviceReset());
    return 0;
}

__global__ void gaussianEliminationRows(double* all_matrixs, int matrix_order)
{
	int matrix_i = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
	int thread_ID = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
	double current_matrix_offset = matrix_i*matrix_order*matrix_order;
    
    int current_row_pivot = 0;
    int current_column_pivot = 0;    
    while(current_row_pivot < matrix_order && current_column_pivot < matrix_order)
    {
        if(thread_ID < matrix_order && thread_ID >= current_column_pivot)                                    
        {
            /* apply gaussian transformation to rows */

            for(int next_row = current_row_pivot + 1; next_row < matrix_order; next_row++)
            {
                int pivot_index_offset = current_row_pivot * matrix_order + current_column_pivot + current_matrix_offset;      /* aii */
                int next_row_index = next_row * matrix_order + current_column_pivot + current_matrix_offset;                   /* aki */
                double factor = all_matrixs[next_row_index] / all_matrixs[pivot_index_offset];                               /* aki / aii */
                int thread_row_index = next_row * matrix_order + thread_ID + current_matrix_offset;                         /* akj */
                int factor_element_index = current_row_pivot * matrix_order + thread_ID + current_matrix_offset;            /* aij */
                all_matrixs[thread_row_index] -= factor * all_matrixs[factor_element_index];                          /* akj = akj - (aki / aii) * aij */
            }
        }

        // wait for all threads in a block to complete and move to the next pivot 
        __syncthreads();
        current_row_pivot++;
        current_column_pivot++;
    }
}
    
    //row => all_matrixes += n * tID
	//column => all_matrixes += tID