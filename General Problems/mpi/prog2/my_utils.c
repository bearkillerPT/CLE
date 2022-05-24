#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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

double calculateSquaredMatrixDeterminant(double** matrix, int matrix_order) {
    int det_sign = gaussianElimination(matrix, matrix_order);
    if (det_sign == 0)
        return 0;
    double det = det_sign;
    for (int i = 0; i < matrix_order; i++)
    {
        det *= matrix[i][i];
    }
    return det;
}