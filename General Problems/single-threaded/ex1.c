#include <stdio.h>
#include <wchar.h>
#include <locale.h>
#include <time.h>

int is_vowel(wint_t char_in)
{
    int special_vowels[30] = {0xC3A1, 0xC381, 0xC3A0, 0xC380, 0xC3A2, 0xC382, 0xC3A3, 0xC383, 0xC3A9, 0xC389, 0xC3A8, 0xC388, 0xC3AA, 0xC38A, 0xC3AD, 0xC38D, 0xC3AC, 0xC38C, 0xC3B3, 0xC393, 0xC3B2, 0xC392, 0xC3B4, 0xC394, 0xC3B5, 0xC395, 0xC3BA, 0xC39A, 0xC3B9, 0xC399};
    if (char_in >= 65 && char_in <= 90 || char_in >= 97 && char_in <= 122) //this line check if you have entered a char_in based on the ascii chart
        if (char_in == 'a' || char_in == 'A' || char_in == 'e' || char_in == 'E' || char_in == 'i' || char_in == 'I' || char_in == 'o' || char_in == 'O' || char_in == 'u' || char_in == 'U')
            return 0; //Vowel
        else {
            for(int special_vowel_i = 0; special_vowel_i < sizeof(special_vowels); special_vowel_i++) {
                if(char_in == special_vowels[special_vowel_i])
                    return 0;
            }
            return 1; //Consonant
        }
    else
        return -1; //Invalid_char
}

int main(int argc, char *argv[])
{
    printf("Running ex1!\n\n");
    setlocale(LC_CTYPE, "pt_PT.UTF-8");
    if (argc == 1)
    {
        printf("Program Usage:\n\t ./ex1 (text*.txt)+");
        return 1;
    }
    for (int text_i = 1; text_i < argc; text_i++)
    {
        clock_t t;
        FILE *text_file = fopen(argv[text_i], "r");
        if (text_file == NULL)
        {
            printf("The text file: %s could not be read!", argv[text_i]);
            return 1;
        }
        t = clock();
        wint_t last_char;
        wint_t current_char;
        int vowel_prefixed_words_count = 0;
        int consonant_sufixed_words_count = 0;
        int words_count = 0;
        while ((current_char = fgetwc(text_file)) != WEOF)
        {
            if (last_char == ' ')
            {
                if (is_vowel(current_char) == 0) 
                    vowel_prefixed_words_count += 1;
            }
            else if (current_char == ' ')
            {
                words_count += 1;
                if (is_vowel(last_char) == 1)
                    consonant_sufixed_words_count += 1;
            }
            last_char = current_char;
        }
        printf("FILE --> %s\n", argv[text_i]);
        printf("#WORDS --> %d\n", words_count);
        printf("#VOWEL_PREFIXED --> %d\n", vowel_prefixed_words_count);
        printf("#CONSONANT_SUFIXED --> %d\n", consonant_sufixed_words_count);
        t = clock() - t;
            double time_taken = ((double)t) / CLOCKS_PER_SEC;
            printf("The program took %f seconds to execute\n\n", time_taken);
        fclose(text_file);
    }
    return 0;
}