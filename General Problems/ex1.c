#include <stdio.h>

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
                if (current_char >= 65 && current_char <= 90 || current_char >= 97 && current_char <= 122) //this line check if you have entered a current_char based on the ascii chart
                {
                    if (current_char == 'a' || current_char == 'A' || current_char == 'e' || current_char == 'E' || current_char == 'i' || current_char == 'I' || current_char == 'o' || current_char == 'O' || current_char == 'u' || current_char == 'U')
                    {
                        vowel_prefixed_words_count += 1;
                    }
                }
            }
            else if (current_char == ' ')
            {
                words_count += 1;
                if (last_char >= 65 && last_char <= 90 || last_char >= 97 && last_char <= 122) //this line check if you have entered a last_char based on the ascii chart
                {
                    if (!(last_char == 'a' || last_char == 'A' || last_char == 'e' || last_char == 'E' || last_char == 'i' || last_char == 'I' || last_char == 'o' || last_char == 'O' || last_char == 'u' || last_char == 'U'))
                    {
                        consonant_sufixed_words_count += 1;
                    }
                }
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