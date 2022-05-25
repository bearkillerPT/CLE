#$1 - the number of workers
#progr1
#cd prog1;
#echo "Compiling Ex1!"
#mpicc -Wall -o ex1 main.c dispatcher.c worker.c;
#echo "Executing Ex1!"
#mpiexec -n $(expr $1 + 1)  ./ex1 text0.txt text1.txt text2.txt text3.txt text4.txt; #--leak-check=full --show-leak-kinds=all
#rm ex1; 
#cd ..;
#progr2
cd prog2;
#echo "Compiling Ex2!"
mpicc -Wall -o ex2 -g ex2.c my_utils.c;
#echo "Executing Ex2!"
mpiexec -n $(expr $1 + 1)  ./ex2 matrixes/mat512_256.bin; #--leak-check=full --show-leak-kinds=all
rm ex2; 
cd ..;
