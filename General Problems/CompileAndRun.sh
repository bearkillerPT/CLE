cmake .. && make 
#&& ./ex1 texts/text0.txt texts/text1.txt texts/text2.txt
valgrind --track-origins=yes ./ex2 matrixs/matrix0.txt matrixs/matrix1.txt matrixs/matrix2.txt