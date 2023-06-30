/*
** Olivia Caulfield
** Cho
** CSC 347
** 3/16/23
*/

/* create random points inside square area with side 1,
* check to see if random point is inside a circle with 
* radius 1. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>  
#include <time.h>


void mcMethod(int iter, float **randPoints, int* inside);

int main(int argc, char **argv){
    printf("Monte Carlo Method: C Code\n");
    // error check for correct number of command line arguments
    if (argc != 2){
        printf("Invalid number of command line arguments.\n");
        exit(100);
    }
    // error check for argument being a positive integer
    int iter = atoi(argv[1]); 
    if (iter <= 0){
        printf("'%s' is not a positive integer.\n", argv[1]);
        exit(101);
    }
    // inside circle and total point pointers
    int* inside = (int*)malloc(sizeof(int));
    int total = iter;

    // define matrix dimensions
    int row = iter; int col = 2;

    // create a 2d array for points
    float **randPoints = (float **) malloc(row * sizeof(float *));
    for(int i = 0; i < row; i++){ randPoints[i] = (float *) malloc(col * sizeof(float)); }
    // fill with random floats
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            randPoints[i][j] = float(rand())/RAND_MAX;
        }
        //printf("(%f, %f)\n", randPoints[i][0], randPoints[i][1]);
    }

    // get start time
    clock_t begin = clock(); 
    // call monte carlo method
    mcMethod(iter, randPoints, inside);
    // get end time
    clock_t end = clock();
    // calculate time for function call
    double time_spent = (double)(end - begin)/ CLOCKS_PER_SEC;
    printf("time spent: %f\n", time_spent);

    // calculate pi
    int in = *inside;
    float pi = float(4 * in )/total;
    printf("pi estimated: %f\n", pi);

    // write time to file
    FILE *fp;
    fp = fopen("pi.csv", "w");
    fprintf(fp, "%f\n", time_spent);
    fclose(fp);

    for (int i = 0; i < row; i++) {free(randPoints[i]);}
    free(randPoints);
    free(inside);
    return(0);
}

void mcMethod(int iter, float **randPoints, int *inside){
    // define row and column size
    int row = iter; int col = 2;
    // new variables
    int x = 0; int y = 1; int count = 0;

    // check if point is inside circle
    for(int i = 0; i < row; i++){
        // use distance formula 
        float distance = randPoints[i][x] * randPoints[i][x] + randPoints[i][y] * randPoints[i][y];
        if(distance <=1){
            count++;
        }
    }
    *inside = count;
}