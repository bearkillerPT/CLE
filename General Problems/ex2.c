#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

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
    printf("%s", res);
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
    if (argc == 1)
    {
        printf("Program Usage:\n\t ./ex2 (matrix*.txt)+");
        return 1;
    }
    for (int matrix_i = 1; matrix_i < argc; matrix_i++)
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

            printf("%s -> %d x %d matrix \n", argv[matrix_i], matrix_length, matrix_length);
            fseek(matrix_file, 0, SEEK_SET);
            read_squared_matrix(matrix_length, matrix, matrix_file);
            print_squared_matrix(matrix_length, matrix);
            printf("\n");
            
            fclose(matrix_file);

            for (int i = 0; i < matrix_length; i++)
                free(matrix[i]);
            free(matrix);

        }
        else 
            printf("The provided file: %s\n couldn't be opened!", argv[matrix_i]);
    }

    return 0;
}