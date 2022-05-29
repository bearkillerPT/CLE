# CLE Assignment 2

## Program 1
### Compiling
```bash
mpicc -Wall -o ex1 main.c dispatcher.c worker.c;
```

### Usage example
To run the program with, for example, 8 workers (and 1 dispatcher) do:
```bash
mpiexec -n 9 ./ex1 text.txt; 
```
## Program 2
### Compiling
```bash
mpicc -Wall -o ex2 ex2.c my_utils.c;
```

### Usage example
To run the program with, for example, 8 workers (and 1 dispatcher) do:
```bash
mpiexec -n 9 ./ex2 mat512_256.bin;
 
#The file is a binary file with an integer describing the number of matrixes in the file followed by another integer, matrix_order, representing the squared matrix order and then followed by matrix_order*matrix_order doubles.
```

