#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
	if (argc == 1)
    {
        printf("Program Usage:\n\t ./ex2 (matrix*.txt)+\n");
        return 1;
    }

	// Error return value
	cudaError_t status;
	// set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
	// Time vars
    clock_t t, t1;
    t = 0;

	//For every file
    for (int matrix_file_i = 1; matrix_file_i < argc; matrix_file_i++)
    {
        FILE *matrix_file = fopen(argv[matrix_file_i], "r");
        if (matrix_file != NULL)
        {
            t1 = clock();
            int matrix_count, matrix_order;
            fread(&matrix_count, sizeof(int), 1, matrix_file);
            fread(&matrix_order, sizeof(int), 1, matrix_file);
			int all_matrixes_size = matrix_count * matrix_order* matrix_order* sizeof(double);
			double* all_matrixes = malloc(all_matrixes_size);

            for (int matrix_i = 0; matrix_i < matrix_count; matrix_i++)
            {
				double *current_matrix = all_matrixes[matrix_i * matrix_order * matrix_order];
				for (int i = 0; i < matrix_order*matrix_order; i++) 
					fread(&current_matrix[i], sizeof(double), 1, matrix_file);
            }
            fclose(matrix_file);
			double *gpu_matrixes;
			CHECK(cudaMalloc((void**) &gpu_matrixes, all_matrixes_size));
			CHECK(cudaMemcpy(gpu_matrixes, all_matrixes, all_matrixes_size, cudaMemcpyHostToDevice);)
			dim3 dimBlock(matrix_order, 1);
			dim3 dimGrid(matrix_count, 1);


			free(all_matrixes);
			CHECK(cudaFree(gpu_matrixes));

        }
        else
            printf("The provided file: %s\n couldn't be opened!\n", argv[matrix_i]);
    }

    //double time_taken = ((double)t) / CLOCKS_PER_SEC;
    //printf("Elapsed time = %lf s\n", time_taken);
    CHECK(cudaDeviceReset());
    return 0;
}

__global__ void scaleAnsSubRow(double* all_matrixs, unsigned int matrix_size)
{
	int tx = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
	int ty; //None?
	int tID = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
	
}
