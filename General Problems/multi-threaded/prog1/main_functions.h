#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <wchar.h>
#include <locale.h>


/* check if is vowel */
int is_vowel(unsigned char c){
    if(c=='a' || c=='e' || c=='i' || c=='o' || c=='u'){
        return 1;
    }
    else if(c=='A' || c=='E' || c=='I' || c=='O' || c=='U'){
        return 1;
    }
    else{
        return 0;
    }
}

/* check if is consonant */

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
/* check if is alpha or underscore  */
int is_alpha_underscore(unsigned char c){
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
        return 1;
    }
    else if((c>='0')&&(c<='9')){
        return 1;
    }
    else if(c=='_'){
        return 1;
    }
    else{
        return 0;
    }
}

/* check if is space, separation or punctiation. 1-> it is */
int is_space_separation_punctuation(unsigned char c){
    if(c==' ' || c =='  ' || c==0xa){ //space
        return 1;
    }
    else if ((c=='-') || (c=='"') || (c=='[')||(c==']')||(c=='(')||(c==')')){ //separation
        return 1;
    }
    else if(c=='.' || c==';' || c == '?' || c =='!' || c == ',' || c==':' || c == 0xE28093 || c == 0xE280A6 ){ //punctuation
        return 1;
    }
    else{
        return 0;
    }
}

/* check if is apostrophe*/
int is_apostrophe(unsigned char c){
    if(c==0x27){
        return 1;
    }
    else{
        return 0;
    }
}

/* get size */
int size_of_array(char *char_array){
    int i = 0;
    while (char_array[i] != NULL) {
        i++;
    }
    return i;
}