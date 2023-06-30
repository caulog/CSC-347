/*
** Olivia Caulfield
** Cho
** CSC 347
** 3/16/23
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>  
#include <time.h>

void histMethod(int numDigits, int piArray[], int *numCount);

int main(int argc, char **argv){
    printf("Distribution of the Digits of Pi: C Code\n");
    int numCount[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    // error check for correct number of command line arguments
    if (argc != 3){
        printf("Invalid number of command line arguments.\n");
        exit(100);
    }
    // File pointer to digits of pi file for CPU memory
    FILE *piFile;
    piFile = fopen(argv[1], "r");
    if(piFile == NULL){
        printf("File '%s' not found\n", argv[1]);
        exit (101);
    }
    // error check for argument being a positive integer
    int numDigits = atoi(argv[2]); 
    if (numDigits <= 0){
        printf("'%s' is not a positive integer.\n", argv[2]);
        exit(102);
    }

    // make array in memory to store numDigits of pi
    int* piArray = (int*) malloc(numDigits*sizeof(int));
    char c;
    for (int i = 0; i < numDigits; i++){
        c = fgetc(piFile);
        if(c == EOF){
            printf("End of file!\n");
            numDigits = i;
            break;
        }
        piArray[i] = ((int)c)-48;
    }

    // get start time
    clock_t begin = clock(); 
    // call monte carlo method
    histMethod(numDigits, piArray, numCount);
    // get end time
    clock_t end = clock();
    // calculate time for function call
    double time_spent = (double)(end - begin)/ CLOCKS_PER_SEC;

    for (int i = 0; i < 10; i++){
        //printf("%d: %f\n", i, (double)numCount[i]/numDigits);
        printf("%d: %d\n", i, numCount[i]);
    }

    printf("time spent: %f\n", time_spent);

    // write time to file
    FILE *fp;
    fp = fopen("piC.csv", "w");
    //fprintf(fp, "%f\n", numCount);
    for (int i = 0; i < 10; i++){
        fprintf(fp, "%f\n", (double)numCount[i]/numDigits*100);
    }
    fclose(fp);
    fclose(piFile);

    free(piArray);
    return(0);
}

void histMethod(int numDigits, int piArray[], int *numCount){
    for (int i = 0; i < numDigits; i++){
        int num = piArray[i];
        numCount[num]++;
        //printf("i: %d\tnum: %d\tcount: %d\n", i, num, numCount[num]);
    }
}