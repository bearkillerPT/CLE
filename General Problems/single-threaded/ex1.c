#include <stdio.h>
#include <time.h>
#include <ctype.h>

/**
 * Test se um determinado char for uma consoante (incluindo caracteres especiais portugueses e caso inferior/superior).
 * não assinados Chars a serem testados
 * @retorno 1 se consonante ou 0 de outra forma
 */
unsigned char is_consonant(unsigned char character)
{
    return((character >= 0x62 && character <= 0x64) ||
           (character >= 0x66 && character <= 0x68) ||
           (character >= 0x6A && character <= 0x6E) ||
           (character >= 0x70 && character <= 0x74) ||
           (character >= 0x76 && character <= 0x7A) ||
           (character >= 0x42 && character <= 0x44) ||
           (character >= 0x46 && character <= 0x48) ||
           (character >= 0x4A && character <= 0x4E) ||
           (character >= 0x50 && character <= 0x54) ||
           (character >= 0x56 && character <= 0x5A));
}

/**
 * Test se um determinado char for uma vogal (incluindo caracteres especiais portugueses e caixa superior/inferior).
 * não assinados Chars a serem testados
 * @retorno 1 se vogal ou 0 de outra forma
 */
unsigned char is_vowel(unsigned char character)
{
    return((character == 0x61 || character == 0x41) ||
           (character == 0x65 || character == 0x45) ||
           (character == 0x69 || character == 0x49) ||
           (character == 0x6f || character == 0x4F) ||
           (character == 0x75 || character == 0x55));
}

/**
 * Test se um determinado char for número ou sublinhado.
 * não assinados Chars a ser testado
 * @retorno 1 se verdadeiro ou 0 de outra forma
 */
unsigned char is_number_underscore(unsigned char character)
{
    return((character >= 0x30 && character <= 0x39) || character == 0x5F);
}

/**
 * @Testar se um determinado char for apóstrofo ou aspas.
 * @não assinados Chars a serem testados
 * @return 1 se verdadeiro ou 0 de outra forma 
 */
unsigned char is_marks(unsigned char character)
{
    return(character == 0x27 || character == 0x98 || character == 0x99);
}

/**
 * Converter um char especial na sua forma base.
 * char a ser testado
 */
void special_character_to_normal(char* character)
{
    if(((unsigned char) *character >= 0xA0 && (unsigned char) *character <= 0xA3) || ((unsigned char) *character >= 0x80 && (unsigned char) *character <= 0x83))
        *character = 0x61;
    if(((unsigned char) *character >= 0xA8 && (unsigned char) *character <= 0xAA) || ((unsigned char) *character >= 0x88 && (unsigned char) *character <= 0x8A))
        *character = 0x65;
    if(((unsigned char) *character >= 0xAC && (unsigned char) *character <= 0xAD) || ((unsigned char) *character >= 0x8C && (unsigned char) *character <= 0x8D))
        *character = 0x69;
    if(((unsigned char) *character >= 0xB2 && (unsigned char) *character <= 0xB5) || ((unsigned char) *character >= 0x92 && (unsigned char) *character <= 0x95))
        *character = 0x6F;
    if(((unsigned char) *character >= 0xB9 && (unsigned char) *character <= 0xBA) || ((unsigned char) *character >= 0x99 && (unsigned char) *character <= 0x9A))
        *character = 0x75;
    if(((unsigned char) *character == 0xA7 || (unsigned char) *character == 0x87))
        *character = 0x63;
}

/**
 * Dado um char, verificar se é whitespace
 * separador, nova linha ou char de retorno de carruagem ou qualquer separador (incluindo a pontuação).
 * Para multibytes, tem em conta apenas o último byte, pelo que o processamento da sequência de multibytes
 * é necessário antes de chamar esta função.
 * unsigned char Char to be tested
 * @return 1 if separator or 0 otherwise
 */
unsigned char is_separator(unsigned char character)
{
    return(character == 0x20 || character == 0x9 || character == 0xA || character == 0xD ||
           character == 0x2D || character == 0x22 || character == 0x9D || character == 0x5B ||
           character == 0x5D || character == 0x28 || character == 0x29 || character == 0x2E ||
           character == 0x2C || character == 0x3A || character == 0x3B || character == 0x3F ||
           character == 0x21 || character == 0x93 || character == 0xA6 || character == 0xAB ||
           character == 0xBB);
}

/**
 * Test se um determinado personagem for um personagem multibyte e consumir bytes extra.
 * Após processamento, o char de entrada terá o último byte da sequência de bytes múltiplos.
 * Chars a serem testados
 * Entrada de fluxo de ficheiros
 */
void process_multibyte(char* character, FILE* file)
{
    if(((unsigned char) *character & 0xF8) == 0xF0)
    {
        // 4 bytes char
        *character = fgetc(file);
        *character = fgetc(file);
        *character = fgetc(file);
    }
    else if(((unsigned char) *character & 0xF0) == 0xE0)
    {
        // 3 bytes char
        *character = fgetc(file);
        *character = fgetc(file);
    }
    else if(((unsigned char) *character & 0xE0) == 0xC0)
    {
        // 2 bytes char
        *character = fgetc(file);
        special_character_to_normal(character);
    }
}

/**
 * Dado um conjunto de ficheiros de texto, conta o número de palavras
 * bem como o número de palavras que começam com uma vogal
 * e número de palavras que terminam com uma consoante para cada ficheiro
 * tendo em conta o alfabeto português.
 * Número de argumentos a partir da linha de comando
 * Entradas de argumentos a partir da linha de comando
 * @return Código de erro de 1 em caso de erro, 0 em caso contrário
 */


int main(int argc, char **argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "Nº de Argumentos insuficientes\n");
        return 1;
    }
    double total_time = 0;
    for(int i = 1; i < argc; i++)
    {
        clock_t timer = clock();
        FILE *file = fopen(argv[i], "r");
        if(file == NULL)
        {
            fprintf(stderr, "Não consegue ler o ficheiro ", argv[i]);
            continue;
        }
        char in_word_flag = 0;
        char in_consonant_flag = 0;
        int word_count = 0;
        int vowel_count = 0;
        int consonant_count = 0;
        char character = 0;
        while((character = getc(file)) != EOF)
        {
            process_multibyte(&character, file);
            // test words
            if(in_word_flag)
            {
                // continuar a ler palavra
                if(is_separator(character))
                {
                    if(in_consonant_flag)
                        consonant_count++;
                    in_consonant_flag = 0;
                    in_word_flag = 0;
                }
                in_word_flag = (is_vowel(character) || is_consonant(character) || is_number_underscore(character) || is_marks(character));
                in_consonant_flag = is_consonant(character) ? 1 : 0;
            }
            else
            {
                // novas palavras
                if(is_vowel(character) || is_consonant(character) || is_number_underscore(character))
                {
                    if(is_vowel(character))
                        vowel_count++;
                    if(is_consonant(character))
                        in_consonant_flag = 1;
                    in_word_flag = 1;
                    word_count++;
                } 
                if(is_separator(character) || is_marks(character))
                {
                    if(in_consonant_flag)
                        consonant_count++;
                    in_consonant_flag = 0;
                    in_word_flag = 0;
                }
            }
        }
        
        if(in_consonant_flag)
            consonant_count++;
        if(in_word_flag)
            word_count++;
        fclose(file);
        fprintf(stdout, "\nCounts for file \"%s\":\n", argv[i]);
        fprintf(stdout, "Words = %d\n", word_count);
        fprintf(stdout, "Words beginning with a vowel = %d\n", vowel_count);
        fprintf(stdout, "Words ending with a consonant = %d\n", consonant_count);
        timer = clock() - timer;
        double elapsed_time = ((double) timer) / CLOCKS_PER_SEC;
        total_time += elapsed_time;
        fprintf(stdout, "Elapsed time: %f seconds\n", elapsed_time);
    }
    fprintf(stdout, "\nTotal elapsed time from all files: %f seconds\n", total_time);
    return 0;
}
