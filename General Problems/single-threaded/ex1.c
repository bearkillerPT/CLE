#include <wchar.h>
#include <stdio.h>
#include <locale.h>
#include <time.h>

int isVowel(wchar_t char_in)
{
    wchar_t special_vowels[30] = {0xC3A1, 0xC381, 0xC3A0, 0xC380, 0xC3A2, 0xC382, 0xC3A3, 0xC383, 0xC3A9, 0xC389, 0xC3A8, 0xC388, 0xC3AA, 0xC38A, 0xC3AD, 0xC38D, 0xC3AC, 0xC38C, 0xC3B3, 0xC393, 0xC3B2, 0xC392, 0xC3B4, 0xC394, 0xC3B5, 0xC395, 0xC3BA, 0xC39A, 0xC3B9, 0xC399};
    if (char_in >= 65 && char_in <= 90 || char_in >= 97 && char_in <= 122)
    { //this line check if you have entered a char_in based on the ascii chart
        if (char_in == 'a' || char_in == 'A' || char_in == 'e' || char_in == 'E' || char_in == 'i' || char_in == 'I' || char_in == 'o' || char_in == 'O' || char_in == 'u' || char_in == 'U')
            return 1; //Vowel
    }
    else
    {
        for (int special_vowel_i = 0; special_vowel_i < 30; special_vowel_i++)
        {
            if ((int)char_in == special_vowels[special_vowel_i])
                return 1;
        }
        return 0; //Not a vowel
    }
    return 0; //Not a vowel
}

int isConsonant(wchar_t char_in)
{
    if (char_in == 0xC387 || char_in == 0xC3A7) // ç || Ç
        return 1;
    if (!isVowel(char_in))
        if (char_in >= 65 && char_in <= 90 || char_in >= 97 && char_in <= 122)
            return 1;
    return 0;
}

int isMergeSymbol(wchar_t char_in)
{

    if (char_in == 0x27 || char_in == 0xE28098 || char_in == 0xE28099)
        return 1;
    return 0;
}

int isPonctuationSymbol(wchar_t char_in)
{
    if ((wchar_t)char_in == '.' || (wchar_t)char_in == ',' || (wchar_t)char_in == ':' || (wchar_t)char_in == ';' || (wchar_t)char_in == '?' || (wchar_t)char_in == '!' || char_in == 0xE28098 || char_in == 0xE280A6)
        return 1;
    return 0;
}

int isSeparationSymbol(wchar_t char_in)
{
    if (char_in == 0x22 || char_in == 0xE2809C || char_in == 0xe2809D || (wchar_t)char_in == '-' || char_in == 0xE28093 || (wchar_t)char_in == '[' || (wchar_t)char_in == ']' || (wchar_t)char_in == '(' || (wchar_t)char_in == ')')
        return 1;
    return 0;
}

int isWhiteSpace(wchar_t char_in)
{
    if (char_in == 0x20 || char_in == 0x9 || char_in == 0xA || char_in == 0xD)
        return 1;
    return 0;
}

int isIrrelevantChar(wchar_t char_in)
{
    return isWhiteSpace(char_in) ||
           isSeparationSymbol(char_in) ||
           isPonctuationSymbol(char_in);
}

//Get a utf8 char!
int fgetutf8c(FILE *f)
{
    int result = 0;
    int input[6] = {};

    input[0] = fgetc(f);
    if (input[0] == EOF)
    {
        // The EOF was hit by the first character.
        result = EOF;
    }
    else if (input[0] < 0x80) //0xxxxxxx
    {
        // the first character is the only 7 bit sequence...
        result = input[0];
    }
    else if ((input[0] & 0xE0) == 0xC0) //10xxxxxx 10xxxxxx
    {
        // This is a 2 byte utf8-char!
        input[1] = fgetc(f);
        result = ((input[0]) << 8) + input[1];
    }
    else if ((input[0] & 0xF0) == 0xE0) //110xxxxx 10xxxxxx 10xxxxxx
    {
        // This is a 3 byte utf8-char!
        input[1] = fgetc(f);
        input[2] = fgetc(f);
        result = (input[0] << 16) + (input[1] << 8) + input[2];
    }
    else if ((input[0] & 0xF8) == 0xF0) //1110xxxx 10xxxxxx 10xxxxxx 10xxxxxx
    {
        // This is a 4 byte utf8-char!
        input[1] = fgetc(f);
        input[2] = fgetc(f);
        input[3] = fgetc(f);
        result = (input[0] << 24) + (input[1] << 16) + (input[2] << 8) + input[3];
    }
    else
        return EOF;
    return result;
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
        wchar_t last_char = ' ';
        wchar_t current_char;
        int vowel_prefixed_words_count = 0;
        int consonant_sufixed_words_count = 0;
        int words_count = 0;
        int merge_symbols_count = 0;
        int last_words_count = 0;
        while ((current_char = fgetutf8c(text_file)) != WEOF)
        {
            if (isMergeSymbol(current_char))
            {
                merge_symbols_count++;
                if (merge_symbols_count % 2 == 0 && words_count == last_words_count+2) {
                    words_count -= 1;
                    printf("last_words_count: %d\n\n", last_words_count);
                    printf("words_count: %d\n", words_count);
                }
                else
                    last_words_count = words_count;
            }
            if (isIrrelevantChar(last_char))
            {
                if (isIrrelevantChar(current_char))
                    continue;
                if (isVowel(current_char))
                    vowel_prefixed_words_count += 1;
                words_count += 1;
            }
            else if (isIrrelevantChar(current_char))
            {

                if (isConsonant(last_char))
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