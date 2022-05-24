/**
 *  \file worker.c
 *
 *  \brief Problem: Assignment 2 - MPI (circular cross-correlation)
 *
 *  Implements all the methods that will be called by the worker.
 *
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <ctype.h>
#include <libgen.h>
#include <string.h>

#include "worker_functions.h"
#include "partfileinfo.h"

/**
 *  \brief Function processDataChunk.
 *
 *  
 *
 * @param buf contains a data chunk - set of chars that form complete words
 * @param partialInfo will store the partial info computed
 */

void processDataChunk(char *buf, PartFileInfo *partialInfo) 
{
    char in_word_flag = 0;
    char in_consonant_flag = 0;

    int buf_size = size_of_array(buf);

    for(int i=0; i<buf_size;i++)
    {
        char converted_char = buf[i];
    

        // process character structure
        if(in_word_flag ==1 )  //inside word
        {
            // still reading a word
            if(is_space_separation_punctuation(converted_char))
            {
                if(in_consonant_flag ==1)
                    (*partialInfo).n_consonants++;
                in_consonant_flag = 0;
                in_word_flag = 0;
            }
            else
            {
                in_word_flag = (is_vowel(converted_char) || is_consonant(converted_char) || 
                                 is_alpha_underscore(converted_char) || 
                                is_apostrophe(converted_char) || is_singlemark(converted_char));
                in_consonant_flag = is_consonant(converted_char);
            }
        }
        else
        {
            // looking for new words
            if(is_vowel(converted_char) || is_consonant(converted_char) || is_alpha_underscore(converted_char))
            {
                if(is_vowel(converted_char))
                    (*partialInfo).n_vowels += 1;
                if(is_consonant(converted_char))
                    in_consonant_flag = 1;
                in_word_flag = 1;
                (*partialInfo).n_words += 1;
            }
            else if(is_space_separation_punctuation(converted_char) || is_apostrophe(converted_char) || is_singlemark(converted_char))
            {
                in_consonant_flag = 0;
                in_word_flag = 0;
            }
        }
    }

}
