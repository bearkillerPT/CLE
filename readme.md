# Large Scale Computation
## General Problems
There are 2 distinct problems to be solved. Firstly a single-threaded program in C then using the following libraries to introduce concurrency:
- pthread
- MPI
- CUDA

The first exercise is to count the amount of words, of words prefixed by a vowel and of word sufixed by a consonant! The examples used to test are:

countWords/text0.txt:
Words = 14

countWords/text1.txt:
Words = 1184

countWords/text2.txt:
Words = 11027

countWords/text3.txt:
Words = 3369

countWords/text4.txt:
Words = 9914

The second exercise is to calculate a squared matrix determinant. The examples used to test are:

| **long version**  | **short version** |
|-------------------|-------------------|
| • mat512_32.bin   | • mat128_32.bin     |
| • mat512_64.bin   | • mat128_64.bin     |
| • mat512_128.bin  | • mat128_128.bin    |
| • mat512_256.bin  | • mat128_256.bin    |

where in matx_y.bin x representes the number of squared matrixs in the file and y the order.
## Results
### Exercise 1:
```
countWords/text0.txt:
Words = 14
Words beginning with a vowel = 10
Words ending with a consonant = 4
Elapsed time: 0.000038 seconds

countWords/text1.txt:
Words = 1184
Words beginning with a vowel = 381
Words ending with a consonant = 365
Elapsed time: 0.000235 seconds

countWords/text2.txt:
Words = 11027
Words beginning with a vowel = 3648
Words ending with a consonant = 3220
Elapsed time: 0.001995 seconds

countWords/text3.txt:
Words = 3369
Words beginning with a vowel = 1004
Words ending with a consonant = 1054
Elapsed time: 0.000622 seconds

countWords/text4.txt:
Words = 9914
Words beginning with a vowel = 3095
Words ending with a consonant = 3175
```
### Exercise 2:
Still in the works.