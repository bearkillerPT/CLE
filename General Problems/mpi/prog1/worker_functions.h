/**
 *  \file worker_functions.h (functions file)
 *
 *  \brief Problem: Assignment 2 - MPI 
 *
 *  Functions used to make the counting of the characters and other calculations in each worker
 *
 *  
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <wchar.h>
#include <locale.h>


/**
 *  \brief Function is_vowel.
 *
 *   check if is vowel
 * 
 *  @param c char to be checked
 *  @return 1 if it is vowel, 0 otherwise.
 *
 */
int is_vowel(unsigned char c)
{
    if(c=='a' || c=='e' || c=='i' || c=='o' || c=='u' || c=='y')
    {
        return 1;
    }
    else if(c=='A' || c=='E' || c=='I' || c=='O' || c=='U' || c=='Y')
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/* check if is consonant | 1-> is consonant */

int is_consonant(unsigned char c){
    if(c=='b' || c=='c' || c=='d' || c=='f' || c=='g' || c=='h'|| c=='j' || c=='k' || c=='l' || c=='m' || c=='n' || c=='p' || c=='q' || c=='r' || c=='s' || c=='t' || c=='v' || c=='w' || c=='x' || c=='y' || c=='z'){
        return 1;
    }
    else if(c=='B' || c=='C' || c=='D' || c=='F' || c=='G' || c=='H'|| c=='J' || c=='K' || c=='L' || c=='M' || c=='N' || c=='P' || c=='Q' || c=='R' || c=='S' || c=='T' || c=='V' || c=='W' || c=='X' || c=='Y' || c=='Z'){
        return 1;
    }
    else{
        return 0;
    }
}
/**
 *  \brief Function is_alpha_underscore.
 *
 *   check if is alpha or underscore
 * 
 *  @param c char to be checked
 *  @return 1 if it is alpha or underscore, 0 otherwise.
 *
 */
int is_alpha_underscore(unsigned char c)
{
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) 
    {
        return 1;
    }
    else if((c>='0')&&(c<='9'))
    {
        return 1;
    }
    else if(c=='_')
    {
        return 1;
    }
    else{
        return 0;
    }
}


/**
 *  \brief Function is_space_separation_punctuation.
 *
 *   check if is space, separation or punctuation. 
 * 
 *  @param c char to be checked
 *  @return 1 if it is space, separation or punctuation, 0 otherwise.
 *
 */
int is_space_separation_punctuation(unsigned char c)
{
    if(c==' ' || c==0xa){ /* space */
        return 1;
    }
    else if ((c=='-') || (c=='"') || (c=='[')||(c==']')||(c=='(')||(c==')')) /* separation */
    { 
        return 1;
    }
    else if(c=='.' || c == ',' || c==':' || c==';' || c == '?' || c =='!') /* punctuation */
    { 
        return 1;
    }
    else{
        return 0;
    }
}

/**
 *  \brief Function is_apostrophe.
 *
 *   check if is apostrophe. 
 * 
 *  @param c char to be checked
 *  @return 1 if it is apostrophe, 0 otherwise.
 *
 */
int is_apostrophe(unsigned char c)
{
    if(c==0x27)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int is_singlemark(unsigned char c)
{
    if ((c==0x81) || (c==0x82) || (c==0x83) ||(c==0x84) ||(c==0x85) ||(c==0x86) ||(c==0x87) ||(c==0x88) || (c==0x89) ||
    (c==0x8a) ||(c==0x8b) ||(c==0x8c) ||(c==0x8d) ||(c==0x8e) ||(c==0x8f) ||(c==0x90) ||(c==0x91) ||(c==0x92) ||
    (c==0x93) ||(c==0x94) ||(c==0x95) ||(c==0x96) ||(c==0x97) ||(c==0x98) ||(c==0x99) ||(c==0x9a) || (c==0x9b) ||
    (c==0x9c) ||(c==0x9d) ||(c==0x9e) ||(c==0x9f) || (c==0xa0) ||(c==0xa1) ||(c==0xa2) ||(c==0xa3) ||(c==0xa4) ||(c==0xa5) ||
    (c==0xa6) ||(c==0xa7) ||(c==0xa8) ||(c==0xa9) || (c==0xaa) ||(c==0xd0) ||(c==0xd1) ||(c==0xd2) || (c==0xd3) ||(c==0xd4) ||
    (c==0xd5) ||(c==0xd6) ||(c==0xd7) ||(c==0xd8) || (c==0xde) || (c==0xe0) || (c==0xe1) ||(c==0xe2) ||(c==0xe3) ||(c==0xe4) ||
    (c==0xe5) || (c==0xe6) ||(c==0xe7) ||(c==0xe8) ||(c==0xe9) || (c==0xea) ||(c==0xeb) ||(c==0xec) ||(c==0xed) || (c==0x72))
    
    { 
        return 1;
    }
    
    else{
        return 0;
    }
}

/**
 *  \brief Function size_of_array.
 *
 *   Returns size of a char array
 * 
 *  @param c array to be checked
 *  @return i, the size of the array
 *
 */
int size_of_array(char *char_array)
{
    int i = 0;
    while (char_array[i] != '\0') 
    {
        i++;
    }
    return i;
}