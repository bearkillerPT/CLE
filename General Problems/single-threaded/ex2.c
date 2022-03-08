#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

void print_squared_matrix(int size, int **matrix)
{
    char number_buffer[20];
    char res[10000];
    strcpy(res, "[[");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            sprintf(number_buffer, "%d", matrix[i][j]);
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

int calculate_squared_matrix_determinant(int size, int **a, int show_decomposition)
{
    if (size == 2)
        return a[0][0] * a[1][1] - a[0][1] * a[1][0];
    else
    {
        int a_det = 0;
        for (int i = 0; i < size; i++)
        {
            int **sub_a = calloc(size - 1, sizeof(int *));
            for (int sub_row = 0; sub_row < size - 1; sub_row++)
                sub_a[sub_row] = calloc(size - 1, sizeof(int));
            for (int a_row = 1; a_row < size; a_row++)
            {
                int sub_a_col = 0;
                for (int a_col = 0; a_col < size; a_col++)
                {
                    if (a_col != i)
                    {
                        sub_a[a_row - 1][sub_a_col] = a[a_row][a_col];
                        sub_a_col++;
                    }
                }
            }
            if (show_decomposition)
            {
                printf("(-1)^%d * %d * \n", i, a[0][i]);
                print_squared_matrix(size - 1, sub_a);
                if (i != size - 1)
                    printf("+\n");
                else
                    printf("\n");
            }
            a_det += pow(-1, i) * a[0][i] * calculate_squared_matrix_determinant(size - 1, sub_a, show_decomposition);
            for (int sub_row = 0; sub_row < size - 1; sub_row++)
                free(sub_a[sub_row]);
            free(sub_a);
        }
        return a_det;
    }
}

int read_squared_matrix(int size, int **a, FILE *matrix_file)
{
    char c = 0;
    int current_i = 0;
    while ((c = fgetc(matrix_file)) != EOF)
    {
        if (c >= 48 && c <= 57)
        {
            a[current_i / size][current_i % size] = (int)c - 48;
            current_i++;
        }
    }
    return 0;
}

int calculate_squared_matrix_legnth(FILE *matrix_file)
{
    char c = 0;
    int numbers_count = 0;
    while ((c = fgetc(matrix_file)) != EOF)
    {
        if (c >= 48 && c <= 57) //If its a number
            numbers_count++;
    }
    double length = sqrt(numbers_count);
    if (length != (int)length)
        return -1;
    else
        return length;
}

int main(int argc, char *argv[])
{
    printf("Running ex2!\n\n");

    if (argc == 1)
    {
        printf("Program Usage:\n\t ./ex2 [--show_decomposition]? (matrix*.txt)+");
        return 1;
    }
    int matrix_i = 1;
    int show_decomposition = 0;
    if (strcmp(argv[1], "--show_decomposition") == 0)
    {
        matrix_i++;
        show_decomposition = 1;
    }
    for (; matrix_i < argc; matrix_i++)
    {
        FILE *matrix_file = fopen(argv[matrix_i], "r");
        if (matrix_file != NULL)
        {
            int matrix_length = calculate_squared_matrix_legnth(matrix_file);

            if (matrix_length == -1)
            {
                printf("The matrix given is not squared (judging by the amount of numbers given)!");
                return 1;
            }
            int **matrix = calloc(matrix_length, sizeof(int *));
            for (int i = 0; i < matrix_length; i++)
                matrix[i] = calloc(matrix_length, sizeof(int));

            printf("\n%s -> %d x %d matrix \n", argv[matrix_i], matrix_length, matrix_length);
            fseek(matrix_file, 0, SEEK_SET);
            read_squared_matrix(matrix_length, matrix, matrix_file);
            print_squared_matrix(matrix_length, matrix);
            if (show_decomposition)
                printf("Solution decompostion:\n");

            clock_t t;
            t = clock();
            printf("Matrix Determinant -> %d\n", calculate_squared_matrix_determinant(matrix_length, matrix, show_decomposition));
            t = clock() - t;
            double time_taken = ((double)t) / CLOCKS_PER_SEC;
            printf("The program took %f seconds to execute\n", time_taken);
            fclose(matrix_file);

            for (int i = 0; i < matrix_length; i++)
                free(matrix[i]);
            free(matrix);
        }
        else
            printf("The provided file: %s\n couldn't be opened!\n", argv[matrix_i]);
    }

    return 0;
}