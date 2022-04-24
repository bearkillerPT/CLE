#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <wchar.h>
#include <locale.h>
#include "sharedRegion_functions.h"


/** \brief file names storage reg */
static char ** fileNamesRegion;

/** \brief file currently processed */
int fileCurrentlyProcessed = 0;

/** \brief total nº of files to process */
static int nFiles;

/** \brief variab to control the 1º of processing each file */
bool firstProcessing = true;



/* construct the chunks */
int MAX_SIZE = 60;
int MAX_BYTES = 10;
int readen_chars = 0;


/** \brief struct to store data of one file*/
typedef struct {
   int  fileId;    /* file with data */  
   int n_words;    /* number words */
   int n_chars;
   int n_consonants;
   int in_word;
   int max_size;
   int max_chars;
   int **counting_array;
   bool done;        
} PARTFILEINFO;

/** \brief to control the position of file reading */
static long pos;

/** \brief flag signaling the next chunk was obtained/processed */
static bool processed;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;

/** \brief workers synchronization point when next chunk was obtained and processed */
static pthread_cond_t process;

/** \brief flag signaling the previously processed partial info was stored */
static bool stored;

/** \brief all partial file infos */
static PARTFILEINFO * partfileinfos;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief workers synchronization point when previously processed partial info was stored in region */
static pthread_cond_t store;




/**
 *  \brief Initialization 
 *
 *  monitor operation.
 */

static void initialization (void) {
    
    processed = false;                                                         /* next chunk was not processed/obtained */
    stored = false;                                                           /* previous partial info not stored*/
    pthread_cond_init (&process, NULL);                                 
    pthread_cond_init (&store, NULL);                                
    setlocale(LC_CTYPE, "");    

}

/**
 *  \brief Store the filenames in the file names region.
 *
 *  Operation carried out by main thread.
 */

void storeFileNames(int nFileNames, char *fileNames[]) {
  
    if ((pthread_mutex_lock (&accessCR)) != 0) {                             /* enter monitor */                       
       perror ("error on entering monitor(CF)");                            
       int status = EXIT_FAILURE;
       pthread_exit(&status);
    }
    
    nFiles = nFileNames;                     /* nº of files */

    fileNamesRegion = malloc(nFiles * sizeof(char*));   /* mem allocation for the region storing the filenames*/

    partfileinfos = (PARTFILEINFO*)malloc(sizeof(PARTFILEINFO) * nFiles);   /* mem allocation for the partial infos per file*/

    for (int i=0; i<nFileNames; i++) {
        fileNamesRegion[i] = malloc((12) * sizeof(char));       /* mem allocation for the filenames*/
        strcpy(fileNamesRegion[i], fileNames[i]);
        partfileinfos[i].done = false;                         
    }

    pthread_once (&init, initialization);    

    if ((pthread_mutex_unlock (&accessCR)) != 0) {                   /* exit monitor */                                                
       perror ("error on exiting monitor(CF)");                     
       int status = EXIT_FAILURE;
       pthread_exit(&status);
    }

}


/**
 *  \brief Obtain next data chunk (currently just one char) of the current file being processed
 *
 *  Operation carried out by workers.
 */

int getDataChunk(int threadId, char *buf, PARTFILEINFO *partialInfo) {

    if ((pthread_mutex_lock (&accessCR)) != 0) {                     /* enter monitor */        
        perror ("error on entering monitor(CF)");                   
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }

    if (firstProcessing == false) {                                  /* no need to wait, no values to process */
        while (stored==false) {                                       /* wait if the previous partial info was no stored */
            if ((pthread_cond_wait (&store, &accessCR)) != 0) {                                                       
                perror ("error on waiting in fifoEmpty");                 
                int status = EXIT_FAILURE;
                pthread_exit (&status);
            }
        }
    }

    if (partfileinfos[fileCurrentlyProcessed].done == true) {     /* no more data to process on file */  
        if (fileCurrentlyProcessed == nFiles - 1) {                /* last file to process*/
            if ((pthread_mutex_unlock (&accessCR)) != 0) {                                                       
                perror ("error on exiting monitor(CF)");                
                int status = EXIT_FAILURE;                              
                pthread_exit(&status);
            }
            return 1;                                                     
        }
        
        fileCurrentlyProcessed++;       /* next file */
        firstProcessing = true;
    }  

    if (firstProcessing == true) {               /* first time process file */
        partfileinfos[fileCurrentlyProcessed].fileId = fileCurrentlyProcessed; 
        partfileinfos[fileCurrentlyProcessed].n_words = 0;
        partfileinfos[fileCurrentlyProcessed].n_chars = 0;
        partfileinfos[fileCurrentlyProcessed].n_consonants = 0;
        partfileinfos[fileCurrentlyProcessed].in_word = 0;
        partfileinfos[fileCurrentlyProcessed].max_size = 60;
        partfileinfos[fileCurrentlyProcessed].max_chars = 0;
        partfileinfos[fileCurrentlyProcessed].counting_array = (int **)calloc(60, sizeof(int *));
		for (int j = 0; j<60; j++){
			partfileinfos[fileCurrentlyProcessed].counting_array[j] = (int *)calloc(j+2, sizeof(int));
		}
    }

    FILE *f = fopen(fileNamesRegion[fileCurrentlyProcessed], "r");

    if (firstProcessing==false) fseek(f, pos, SEEK_SET );  /* position where stopped read last time */
    if (firstProcessing==true) firstProcessing = false;

    wchar_t c;
    c = fgetwc(f);    /* get next char */
    pos = ftell(f);   /* current position of file reading */

   
    char converted_char = convert_multibyte(c);
  
    /* nº of chars < MAX_BYTES ----> buffer*/
    if(readen_chars<MAX_BYTES){
        buf[readen_chars] = converted_char;
        readen_chars++;
    }
    /* we use the array MAX_SIZE ( the char is not end of word) || the char is end of word - buffer emptied and another word start
    */
    else{
        if(is_end_of_word(converted_char) == 0){
            buf[readen_chars] = converted_char;
            readen_chars++;
        }
        else{
            memset(buf, 0, MAX_BYTES+MAX_SIZE);
            readen_chars = 0;
            buf[readen_chars] = converted_char;
            readen_chars++;
        }
    }

    fclose(f);
    
    if ( c == WEOF)  { /*last character of current file */
        partfileinfos[fileCurrentlyProcessed].done = true;   /* done processing current file */
    }

    *partialInfo = partfileinfos[fileCurrentlyProcessed];

    processed = true;     /* obtained chunk */
    if ((pthread_cond_signal (&process)) != 0) {      /* worker know that the next chunk has been obtainedd */                                                                                                                         
        perror ("error on waiting in fifoEmpty");                 
        int status = EXIT_FAILURE;
        pthread_exit (&status);
    }
    stored = false; 

    if ((pthread_mutex_unlock (&accessCR)) != 0) {                                                  
        perror ("error on exiting monitor(CF)");                                        
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }

    return 0;
}


/**
 *  \brief Save partial results.
 *
 *  Done by Workers
 */

void savePartialResults(int threadId, PARTFILEINFO *partialInfo) {

    if ((pthread_mutex_lock (&accessCR)) != 0) {                           
        perror ("error on entering monitor(CF)");                   
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }

    while (processed == false) {                                               /* wait if the next chunk was not obtained/processed */
        if ((pthread_cond_wait (&process, &accessCR)) != 0) {
            perror ("error on waiting in fifoEmpty");                  
            int status = EXIT_FAILURE;
            pthread_exit (&status);
        }
    }

    partfileinfos[fileCurrentlyProcessed] = *partialInfo;                   /* save partial info */
    
    stored = true;                                /* new partial info saved */
    if ((pthread_cond_signal (&store)) != 0) {
        perror ("error on waiting in fifoEmpty");                  
        int status = EXIT_FAILURE;
        pthread_exit (&status);
    }
    processed = false;                   /* next chunk was not processed yet */
     
    if ((pthread_mutex_unlock (&accessCR)) != 0) {                                                   
        perror ("error on exiting monitor(CF)");                                        
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }

}


/**
 *  \brief Print final results.
 *
 *  Done by Main
 */

void printProcessingResults() {

    if ((pthread_mutex_lock (&accessCR)) != 0) {                        
        perror ("error on entering monitor(CF)");                   
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }

    for (int i=0; i<nFiles; i++) {                 

        printf("\nFile name: %s\n", fileNamesRegion[i]);

        printf("Total number of words = %d\n", partfileinfos[i].n_words);
        //printf("Total number of words starting with a vowel = %d\n", partfileinfos[i]);
        //printf("Total number of words ending with a consonant = %d\n", partfileinfos[i].n_words);

        
    }

    if ((pthread_mutex_unlock (&accessCR)) != 0) {                                                  
        perror ("error on exiting monitor(CF)");                                        
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }

}
