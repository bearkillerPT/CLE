# Cuda Gaussian Elimination
The gaussian elimination was solved by:
- Rows
- Columnns

The input files have two ints:
- m: Matrices Counts
- n: Matrix order (squared matrices)
followed by m * n * n doubles containing the coeficientes of each matrix.  

Each gaussian elimination is supposed to be called with m blocks each containing n threads.

## Elimination by rows
If diagonal pivot is 0 the the alg finds a row with non 0 coeficient to swap with. It then performs the factorization by row, each thread calculating each column's coeficient until the pivot (the rest are 0s),
Compiling and running:
``` bash
make;
./RowsGaussianDet mat512_256.bin
```

## Elimination by columns
If diagonal pivot is 0 the the alg finds a column with non 0 coeficient to swap with. It then performs the factorization by column, each thread calculating each row's coeficient until the pivot (the rest are 0s),
Compiling and running:
``` bash
make;
./ColsGaussianDet mat512_256.bin
```