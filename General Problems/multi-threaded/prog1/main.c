#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

#include "sharedRegion.h"
#include "main_functions.h"

/** variables to define the size of the buffer*/
int MAX_SIZE_WORD = 60;
int MAX_BYTES_TO_READ = 12;

/** \brief worker life cycle routine */
static void *worker(void *id);

/**
 *  \brief Main thread.
 *
 *  Store the filenames in the shared region;  create the workers and
 *  waiting for their termination; print the results;
 *
 */

int main(int argc, char *argv[])
{

    int nThreads = atoi(argv[1]); /* nº of workers to create */

    char *filenames[argc - 2]; /* file names */

    for (int i = 0; i < argc; i++)
        filenames[i] = argv[i + 2];

    struct timespec tStart, tStop;

    clock_gettime(CLOCK_MONOTONIC, &tStart); /* start counting time of execution */

    pthread_t tIdWork[nThreads]; /* workers internal thread id array */
    unsigned int work[nThreads]; /* workers defined thread id array */
    int *status_p;               /* pointer to execution status */

    int t;

    for (int t = 0; t < nThreads; t++)
        work[t] = t;

    storeFileNames(argc - 2, filenames);

    printf("File names stored in the shared region.\n");

    for (t = 0; t < nThreads; t++)
    {
        if (pthread_create(&tIdWork[t], NULL, worker, &work[t]) != 0) /* create thread worker */
        {
            perror("error on creating thread producer");
            exit(EXIT_FAILURE);
        }
    }

    printf("Workers created and start processing.\n");

    for (t = 0; t < nThreads; t++)
    {
        if (pthread_join(tIdWork[t], (void *)&status_p) != 0) /* wait for thread worker */
        {
            perror("error on waiting for thread producer");
            exit(EXIT_FAILURE);
        }
        printf("thread worker, with id %u, has terminated. \n", t);
    }

    printProcessingResults(); /* results */

    printf("Terminated.\n");

    clock_gettime(CLOCK_MONOTONIC, &tStop); /* end of execution time */

    double time_taken = (tStop.tv_sec - tStart.tv_sec);
    time_taken += (tStop.tv_nsec - tStart.tv_nsec) / 1000000000.0;
    printf("\nElapsed time = %.6f s\n", time_taken);

    return 0;
}

/**
 *  \brief Function processDataChunk
 */

void processDataChunk(char *buf, PARTFILEINFO *partialInfo)
{
    char converted_char;
    int buf_size = size_of_array(buf);
    for (int i = 0; i < buf_size; i++)
    {
        converted_char = buf[i];
    }

    if (!(*partialInfo).in_word)
    {
        if (is_alpha_underscore(converted_char)) //Word that begins with a vowel
        {

            (*partialInfo).in_word = 1;
            (*partialInfo).n_words++;
            (*partialInfo).n_chars++;
            (*partialInfo).n_consonants = (*partialInfo).n_consonants + !is_vowel(converted_char);
        }
        else if (is_apostrophe(converted_char) || is_space_separation_punctuation(converted_char))
        {
            return;
        }
    }
    else
    {
        if (is_alpha_underscore(converted_char))
        {
            (*partialInfo).n_chars++;
            (*partialInfo).n_consonants = (*partialInfo).n_consonants + !is_vowel(converted_char);
        }
        else if (is_apostrophe(converted_char))
        {
            return;
        }
        else if (is_space_separation_punctuation(converted_char)) //Word that ends with a consonant
        { 

            printf("buf - %s\n", buf);
            printf("converted_char - %c\n", converted_char);
            (*partialInfo).in_word = 0;
            (*partialInfo).counting_array[(*partialInfo).n_chars - 1][(*partialInfo).n_consonants]++;
            if ((*partialInfo).n_chars > (*partialInfo).max_chars)
            {
                (*partialInfo).max_chars = (*partialInfo).n_chars;
            }
            (*partialInfo).n_chars = 0;
            (*partialInfo).n_consonants = 0;
        }
    }
}

/**
 *  \brief Function worker.
 *
 *  Get data chunks, process them and store the results in shared region.
 *
 */

static void *worker(void *par)
{

    unsigned int id = *((unsigned int *)par); /* worker id */

    /* buffer has size  = MAX_BYTES_TO_READ bytes + MAX_SIZE_WORD 
    Will be a set of complete words. */
    char buf[MAX_BYTES_TO_READ + MAX_SIZE_WORD];

    PARTFILEINFO partialInfo; /* struct to store partial info */

    while (getDataChunk(id, buf, &partialInfo) != 1)
    {                                         
        processDataChunk(buf, &partialInfo);  
        savePartialResults(id, &partialInfo); 
    }

    int status = EXIT_SUCCESS;
    pthread_exit(&status);
}
