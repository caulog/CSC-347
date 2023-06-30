/* 
**  Olivia Caulfield
**  Cho
**  CSC 347
**  2/28/23
*/

#include <stdio.h> 
#include <stdlib.h>
#include <errno.h>  
#include <time.h>

void matrixMult(float **m1, float **m2, float **m3, int row, int col);

int main(int argc, char **argv){
    printf("Matrix Multiplication: C Code\n");
    
    // error check for correct number of command line arguments
    if (argc != 2){
        printf("Invalid number of command line arguments.\n");
        exit(100);
    }
    // error check for argument being a positive integer
    int row = atoi(argv[1]); 
    int col = atoi(argv[1]);
    if (row <= 0){
        printf("'%s' is not a positive integer.\n", argv[1]);
        exit(101);
    }

    // create matricies filled with random float 
    int max = 1;

    // allocate matrix 1 memory
    float **m1 = (float **) malloc(row * sizeof(float *));
    for(int i = 0; i < row; i++){ m1[i] = (float *) malloc(col * sizeof(float)); }
    // allocate matrix 2 memory
    float **m2 = (float **) malloc(row * sizeof(float *));
    for(int i = 0; i < row; i++){ m2[i] = (float *) malloc(col * sizeof(float)); }

    // fill matrix 1 and 2
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            m1[i][j] = ((float)rand()/(float)(RAND_MAX)) * max;
            m2[i][j] = ((float)rand()/(float)(RAND_MAX)) * max;
        }
    }

    // allocate matrix 3 memory
    float **m3 = (float **) malloc(row * sizeof(float *));
    for(int i = 0; i < row; i++){ m3[i] = (float *) malloc(col * sizeof(float)); }

    // get start time
    clock_t begin = clock(); 
    // multiply the matricies
    matrixMult(m1, m2, m3, row, col);
    // get end time
    clock_t end = clock();
    // calculate time to multiply
    double time_spent = (double)(end - begin)/ CLOCKS_PER_SEC;

    // print time to multiply
    printf("time: %f s\n", time_spent);

    // open a file for write
    FILE *fp;
    fp = fopen("product.dat", "w");

    // write product to a file
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            fprintf(fp, "%f\t", m3[i][j]);
        }
        fprintf(fp, "\n");
    }

    free(m1);
    free(m2);
    free(m3);
    fclose(fp);

    return(0);
}

void matrixMult(float **m1, float **m2, float **m3, int row, int col){
    // multipy each value and sum with corresponding values
    // store in output matrix
    int out = row;
    for (int i = 0; i < out; i++){
        for(int j = 0; j < row; j++){
            for(int k = 0; k < col; k++){
                m3[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}