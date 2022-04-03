#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

void printSquaredMatrix(double **matrix, int size)
{
    char number_buffer[20];
    char res[10000];
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            sprintf(number_buffer, "%lf ", matrix[i][j]);
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
    return -1;
}

int gaussianElimination(double **a, int size)
{ // returns the determinant sign (+det / -det)
    int res_sign = 1;
    for (int i = 0; i < size - 1; i++)
    {
        if (a[i][i] == 0)
        {
            int swap_res = findAndSwapCols(a, size, i);
            if (swap_res == 0)
                return 0;
            res_sign *= swap_res;
        }
        for (int k = i + 1; k < size; k++)
        {
            for (int j = i; j < size; j++)
            {
                a[k][j] -= (a[k][i] / a[i][i]) * a[i][j];
            }
        }
    }
    return res_sign;
}

double readSquaredMatrixAndCalculateDeterminant(FILE *matrix_file, int matrix_order)
{
    double **matrix = calloc(matrix_order, sizeof(double *));
    for (int i = 0; i < matrix_order; i++)
    {
        matrix[i] = calloc(matrix_order, sizeof(double));
        for (int j = 0; j < matrix_order; j++)
            fread(&matrix[i][j], sizeof(double), 1, matrix_file);
    }
    int det_sign = gaussianElimination(matrix, matrix_order);
    if (det_sign == 0)
        return 0;
    double det = 1;
    for (int i = 0; i < matrix_order; i++)
    {
        det *= matrix[i][i];
    }
    for (int i = 0; i < matrix_order; i++)
        free(matrix[i]);
    free(matrix);
    return det_sign * det;
}

int main(int argc, char *argv[])
{
    printf("Running ex2!\n\n");

    if (argc == 1)
    {
        printf("Program Usage:\n\t ./ex2 (matrix*.txt)+");
        return 1;
    }
    clock_t t, t1;
    t = 0;
    int matrix_i = 1;
    for (; matrix_i < argc; matrix_i++)
    {
        FILE *matrix_file = fopen(argv[matrix_i], "r");
        if (matrix_file != NULL)
        {
            printf("Processing matrix %d\n", matrix_i);
            t1 = clock();

            int matrix_count, matrix_order;
            fread(&matrix_count, sizeof(int), 1, matrix_file);
            fread(&matrix_order, sizeof(int), 1, matrix_file);
            for (int matrix_i = 0; matrix_i < matrix_count; matrix_i++)
            {
                printf("%d\n", matrix_count);
                printf("%d\n", matrix_order);
                double det = readSquaredMatrixAndCalculateDeterminant(matrix_file, matrix_order);
                t += clock() - t1;
                printf("The determinant is %E\n", det);
                return 0;
            }
            double time_taken = ((double)t) / CLOCKS_PER_SEC;
            printf("Elapsed time = %lf s\n", time_taken);
            fclose(matrix_file);
        }
        else
            printf("The provided file: %s\n couldn't be opened!\n", argv[matrix_i]);
    }

    return 0;
}