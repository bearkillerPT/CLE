#$1 - the number of workers
echo ;
mpicc -Wall -o ex2 -g ex2.c my_utils.c;
mpiexec -n $(expr $1 + 1)  ./ex2 matrixes/mat128_256.bin matrixes/mat128_128.bin; #--leak-check=full --show-leak-kinds=all
rm ex2; 