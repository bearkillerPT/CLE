#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

void printSquaredMatrix(double **matrix, int size)
{
    char number_buffer[20];
    char res[10000];
    strcpy(res, "[[");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            sprintf(number_buffer, "%lf", matrix[i][j]);
            strcat(res, number_buffer);

            if (j != size - 1)
                strcat(res, ",");
        }
        if (i != size - 1)
            strcat(res, "],\n[");
        else
            strcat(res, "]");
    }
    strcat(res, "]");
    printf("%s\n", res);
    fflush(stdout);
}

int findAndSwapZeroCoefCol(double **a, int size, int col_in)
{
    int *first_col = calloc(size, sizeof(int));
    int *second_col = calloc(size, sizeof(int));
    int second_col_index = size;
    for (int i = 0; i < size; i++)
    {
        first_col[i] = a[i][col_in];
        if (i > col_in && a[col_in][i] == 0)
            second_col_index = i;
    }
    if (second_col_index == size)
        return 0;
    for (int i = 0; i < size; i++)
        first_col[i] = a[i][second_col_index];
    for (int i = 0; i < size; i++)
    {
        a[i][col_in] = second_col[i];
        a[i][second_col_index] = first_col[i];
    }
    return -1;
}

int gaussianElimination(double **a, int size)
{ //returns the determinant sign (+det / -det)
    int res_sign = 1;
    for (int i = 0; i < size - 1; i++)
    {
        if (a[i][i] == 0)
        {
            int swap_res = findAndSwapZeroCoefCol(a, size, i);
            if (swap_res == 0)
                return 0;
            res_sign *= swap_res;
        }
        for (int k = i + 1; k < size; k++)
        {
            for (int j = i; j < size; j++)
            {
                a[k][j] -= a[i][j] * a[k][i] / a[i][i];  
            }
        }
    }
    int res = 1;
    for (int i = 0; i < size - 1; i++)
    res *= a[i][i];
    res *= res_sign;
    return res;
}
double calculateSquareMatrixDeterminant(double **a, int size)
{
    gaussianElimination(size, a);

    return 0;
}

int readSquaredMatrixAndCalculateDeterminants(FILE *matrix_file)
{
    int matrix_count = 0;
    int current_matrix_index = 0;
    int matrix_order = 0;
    fread(&matrix_count, 1, sizeof(int), matrix_file);
    fread(&matrix_order, 1, sizeof(int), matrix_file);
    printf("%d %dx%d matrixs\n", matrix_count, matrix_order, matrix_order);
    fflush(stdout);
    double a = 0; //matrix entrie
    int current_i = 0;

    double **matrix = calloc(matrix_order, sizeof(double *));
    for (int i = 0; i < matrix_order; i++)
        matrix[i] = calloc(matrix_order, sizeof(double));

    while ((fread(&a, 1, sizeof(double), matrix_file)) == sizeof(double))
    {
        matrix[current_i / matrix_order][current_i % matrix_order] = a;
        current_i++;
        if (current_i == matrix_order * matrix_order)
        {
            current_matrix_index++;
            current_i = 0;
            printf("Matrix_%d:%d %dx%d. Det -> %f\n", current_matrix_index, matrix_count, matrix_order, matrix_order, calculateSquareMatrixDeterminant(matrix, matrix_order));
            printSquaredMatrix(matrix, matrix_order);
            return 1;
        }
    }

    for (int i = 0; i < matrix_order; i++)
        free(matrix[i]);
    free(matrix);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("Running ex2!\n\n");

    if (argc == 1)
    {
        printf("Program Usage:\n\t ./ex2 (matrix*.txt)+");
        return 1;
    }
    int matrix_i = 1;
    for (; matrix_i < argc; matrix_i++)
    {
        FILE *matrix_file = fopen(argv[matrix_i], "r");
        if (matrix_file != NULL)
        {

            clock_t t;
            t = clock();
            readSquaredMatrixAndCalculateDeterminants(matrix_file);
            t = clock() - t;
            double time_taken = ((double)t) / CLOCKS_PER_SEC;
            printf("The program took %f seconds to execute\n", time_taken);
            fclose(matrix_file);
        }
        else
            printf("The provided file: %s\n couldn't be opened!\n", argv[matrix_i]);
    }

    return 0;
}