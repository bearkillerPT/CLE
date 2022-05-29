/**
 *  \file partfileinfo.h
 *
 *  \brief Problem: Assignment 2 - MPI 
 *
 *  Struct used to store partial infos.
 *
 *  
 */

#include <stdbool.h>

#ifndef PARTFILEINFO_H
#define PARTFILEINFO_H

/** \brief struct to store data of one file*/
typedef struct {
   int  fileId;    /* file with data */  
   int n_words;    /* number words */
   int n_vowels;  /*number of vowels*/
   int n_consonants;    /* number consonants */
   int in_word;     /* to control the reading of a word */
   int counting_array[50][51];    /*  to store and process the final countings  -> counting_array[MAX_SIZE_WORD][MAX_SIZE_WORD+1]*/
   bool firstProcessing; /* indicates wether it is the first time processing that file or not */
   bool done;        /* to control the end of processing */ 
} PartFileInfo;

#endif