# CLE Assignment 1 

## Compiling

```c
gcc -Wall -o main main.c sharedRegion.c -lpthread -lm
```

## Usage examples

```c
./main numberWorkers [files]

./main 1 text0.txt
./main 4 text0.txt
./main 4 text0.txt text1.txt text2.txt text3.txt text4.txt
```

