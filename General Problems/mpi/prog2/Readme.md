# CLE Assignment 2

## Compiling
```c
mpicc -Wall -o ex2 ex2.c my_utils.c;
```

## Usage examples

```c
./ex2 [files]

mpiexec -n 8 ./ex2 mat128_256.bin; 
//The file is a binary file with an integer describing the number of matrixes in the file followed by another integer, matrix_order, representing the squared matrix order and then followed by matrix_order*matrix_order doubles.
```

