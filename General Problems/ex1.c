#include <stdio.h>

int is_vowel(char char_in)
{
    if (char_in >= 65 && char_in <= 90 || char_in >= 97 && char_in <= 122) //this line check if you have entered a char_in based on the ascii chart
        if (char_in == 'a' || char_in == 'A' || char_in == 'e' || char_in == 'E' || char_in == 'i' || char_in == 'I' || char_in == 'o' || char_in == 'O' || char_in == 'u' || char_in == 'U')
            return 0; //Vowel
        else
            return 1; //Consonant
    else
        return -1; //Invalid_char
}

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        printf("Program Usage:\n\t ./ex1 (text*.txt)+");
        return 1;
    }
    for (int text_i = 1; text_i < argc; text_i++)
    {
        FILE *text_file = fopen(argv[text_i], "r");
        if (text_file == NULL)
        {
            printf("The text file: %s could not be read!", argv[text_i]);
            return 1;
        }
        char last_char;
        char current_char;
        int vowel_prefixed_words_count = 0;
        int consonant_sufixed_words_count = 0;
        int words_count = 0;
        while ((current_char = fgetc(text_file)) != EOF)
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
        printf("#CONSONANT_SUFIXED --> %d", consonant_sufixed_words_count);
        if (text_i != argc - 1)
            printf("\n");
        printf("\n");
        fclose(text_file);
    }
    return 0;
}