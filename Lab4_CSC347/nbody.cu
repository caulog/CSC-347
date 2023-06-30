// ./a.out # > nbody.pdb direct the output to a file

#include <cstdio>
#include <stdio.h> 
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

//#define N 9999     // number of bodies
#define MASS 0     // row in array for mass
#define X_POS 1    // row in array for x position
#define Y_POS 2    // row in array for y position
#define Z_POS 3    // row in array for z position
#define X_VEL 4    // row in array for x velocity
#define Y_VEL 5    // row in array for y velocity
#define Z_VEL 6    // row in array for z velocity
#define G 200      // "gravitational constant" (not really)
#define BLOCK_DIM 32

__global__ void force_calc(float* body, float* Fx_dir, float* Fy_dir, float* Fz_dir, int N);
__global__ void pos_update(float* body, float* Fx_dir, float* Fy_dir, float* Fz_dir, float dt, int N);


void crossProduct(float vect_A[], float vect_B[], float cross_P[]) { 
  cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1]; 
  cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2]; 
  cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0]; 
}

__host__ __device__ void norm(float &x, float &y, float &z) {
  float mag = sqrt(x*x+y*y+z*z);
  x/=mag; y/=mag; z/=mag;
}

   ///////////////////////////////////////////////////////////////////////////
  // TO DETERMINE WHAT IS ABLE TO BE PARALELIZED, START WITH THE FOR LOOPS //
 ///////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  FILE *fp;
  fp = fopen("nbody_gpu.csv", "w");

  float dt = 0.05; // time interval
  float body[10000][7]; // data array of bodies -> each body with 7 characteristics
  
  int N = 9999; // comment out if running for loop
//for(int N = 99; N < 9999; N = N + 100){
  double total_s = 0.0;
  double total_p = 0.0;

  clock_t begins1 = clock(); 

  int numBody = 10000*7;
  float *dev_body;
  cudaMalloc((void**) &dev_body, sizeof(float)*numBody);
  
  int tmax = 0; 

  // error check for correct number of command line arguments
  if (argc != 2){
    printf("Invalid number of command line arguments.\n");
    exit(100);
  }
  // error check for argument being a positive integer
  tmax = atoi(argv[1]);
  if (tmax <= 0){
    printf("'%s' is not a positive integer.\n", argv[1]);
    exit(101);
  }

  // initalize memory on CPU
  float Fx_dir[N]; // force in x direction
  float Fy_dir[N]; // force in y direction
  float Fz_dir[N]; // force in z direction
  // initialize memory on GPU
  int size = N * sizeof(float);
  float *dev_Fx_dir; cudaMalloc((void**) &dev_Fx_dir, size);
  float *dev_Fy_dir; cudaMalloc((void**) &dev_Fy_dir, size);
  float *dev_Fz_dir; cudaMalloc((void**) &dev_Fz_dir, size);

  // define grid and block dim
  dim3 dimBlock(BLOCK_DIM); 
  dim3 dimGrid((N+BLOCK_DIM-1)/BLOCK_DIM);
  // assign each body a random initial positions and velocities
  srand48(time(NULL));
  float vect_A[3];
  float vect_B[3];
  float cross_P[3];

  // black hole at the center
  body[0][MASS] = 4000.0;
  body[0][X_POS] = 0.0;
  body[0][Y_POS] = 0.0;
  body[0][Z_POS] = 0.0;
  body[0][X_VEL] = 0.0;
  body[0][Y_VEL] = 0.0;
  body[0][Z_VEL] = 0.0;
  
  // loop through N bodies, set initial position, and set initial velocities
  for (int i = 1; i < N; i++) {
    body[i][MASS] = 0.001; // initialize the mass of the other bodies to be 0.001

    // DONE: initial coordinates centered on origin, ranging -150.0 to +150.0
    body[i][X_POS] = drand48() * 300 - 150;
    body[i][Y_POS] = drand48() * 300 - 150;
    body[i][Z_POS] = drand48() * 300 - 150;

    // initial velocities directions around z-axis
    vect_A[0]= body[i][X_POS];
    vect_A[1]= body[i][Y_POS];
    vect_A[2]= body[i][Z_POS];
    norm(vect_A[0], vect_A[1], vect_A[2]);
    vect_B[0]= 0.0; vect_B[1]= 0.0; vect_B[2]= 1.0;
    cross_P[0] = 0.0; cross_P[1] = 0.0; cross_P[2] = 0.0; 
    crossProduct(vect_A, vect_B, cross_P);

    // random initial velocities magnitudes
    body[i][X_VEL] = drand48() * 100 * cross_P[0];
    body[i][Y_VEL] = drand48() * 100 * cross_P[1];
    body[i][Z_VEL] = drand48() * 100 * cross_P[2];
  }
  cudaMemcpy(dev_body, body, numBody*sizeof(float), cudaMemcpyHostToDevice);

  // print out initial positions in PDB format
  printf("MODEL %8d\n", 0);
  for (int i = 0; i < N; i++) { printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n", "ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00); }
  printf("TER\nENDMDL\n");

  clock_t ends1 = clock();
  double s1 = (double)(ends1 - begins1)/ CLOCKS_PER_SEC;
  total_s += s1;

  // step through each time step
  cudaEvent_t startp1;
  cudaEventCreate(&startp1);
  cudaEvent_t stopp1;
  cudaEventCreate(&stopp1);

  for (int t = 0; t < tmax; t++) {
    // force calculation
    clock_t begins2 = clock(); 
    // DONE: initialize forces to zero
    for (int i = 0; i < N; i++) {
      Fx_dir[i] = 0; // force in x direction
      Fy_dir[i] = 0; // force in y direction
      Fz_dir[i] = 0; // force in z direction
    }
    clock_t ends2 = clock();
    double s2 = (double)(ends2 - begins2)/ CLOCKS_PER_SEC;
    total_s += s2;

    ///////////// parallel code /////////////////
    // start timer
    cudaEventRecord(startp1,0);

    // copy relevant code for force calculation
    cudaMemcpy(dev_body, body, numBody*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Fx_dir, Fx_dir, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Fy_dir, Fy_dir, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Fz_dir, Fz_dir, size, cudaMemcpyHostToDevice);

    // calculate forces in body x due to all other bodies in kernel
    force_calc<<<dimGrid, dimBlock>>>(dev_body, dev_Fx_dir, dev_Fy_dir, dev_Fz_dir, N);
    cudaDeviceSynchronize();

    // copy back to host
    cudaMemcpy(body, dev_body, numBody*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Fx_dir, dev_Fx_dir, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Fy_dir, dev_Fy_dir, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Fz_dir, dev_Fz_dir, size, cudaMemcpyDeviceToHost);

    // copy relevant code for position updates
    cudaMemcpy(dev_body, body, numBody*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Fx_dir, Fx_dir, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Fy_dir, Fy_dir, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Fz_dir, Fz_dir, size, cudaMemcpyHostToDevice);

    // update postions and velocity in array
    pos_update<<<dimGrid,dimBlock>>>(dev_body, dev_Fx_dir, dev_Fy_dir, dev_Fz_dir, dt, N);
    cudaDeviceSynchronize();
    
    // copy back to host
    cudaMemcpy(body, dev_body, numBody*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Fx_dir, dev_Fx_dir, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Fy_dir, dev_Fy_dir, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Fz_dir, dev_Fz_dir, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stopp1,0);
    cudaEventSynchronize(stopp1);
    float diffp1 = 0.0;
    cudaEventElapsedTime(&diffp1, startp1, stopp1);
    //fprintf(stderr, "%f\n", diffp1/1000);
    total_p += (diffp1/1000);

    clock_t begins3 = clock(); 
    // print out positions in PDB format
    printf("MODEL %8d\n", t+1);
    for (int i = 0; i < N; i++) { printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n", "ATOM", i+1, "CA ", "GLY", "A", i+1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00); }
    printf("TER\nENDMDL\n");
    clock_t ends3 = clock();
    double s3 = (double)(ends3 - begins3)/ CLOCKS_PER_SEC;
    total_s += s3;

  }  // end of time period loop ////////// THIS ///////// CAN //////// BE ///////// PARALELLIZED
  
  // deallocate timers
  cudaEventDestroy(startp1);
  cudaEventDestroy(stopp1);

  fprintf(fp, "%d, %f, %f\n", N, total_s, total_p);

  cudaFree(dev_body);
  cudaFree(dev_Fx_dir);
  cudaFree(dev_Fy_dir);
  cudaFree(dev_Fz_dir);
  //}

  fclose(fp);
}

__global__ void force_calc(float* body, float* Fx_dir, float* Fy_dir, float* Fz_dir, int N){
  // get thread
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  //printf("%d calc called\n", x);

  // make sure x is in bounds
  if(x < N){
    for (int i = 0; i < N; i++) {   // all other bodies
	    // position differences in x-, y-, and z-directions
	    float x_diff, y_diff, z_diff;

	    if (i != x) {
	      // DONE: calculate position difference between body i and x in x-,y-, and z-directions
        x_diff = body[i*7+X_POS] - body[x*7+X_POS];
        y_diff = body[i*7+Y_POS] - body[x*7+Y_POS];
        z_diff = body[i*7+Z_POS] - body[x*7+Z_POS];
	  
	      // calculate distance (r)
	      float rr = (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
	      float r = sqrt(rr);

	      // force between bodies i and x
        float F = 0;

	      // if sufficiently far away, apply gravitation force
	      if (r > 50.0) {
	        // DONE: compute gravitational force between body i and x
          F = -1.0 * (G * body[i*7+MASS] * body[x*7+MASS])/rr;

          norm(x_diff, y_diff, z_diff);

          atomicAdd(&Fx_dir[x], x_diff * F);
          atomicAdd(&Fy_dir[x], y_diff * F);
          atomicAdd(&Fz_dir[x], z_diff * F);
	        }
	      }
      }
  }
}


__global__ void pos_update(float* body, float* Fx_dir, float* Fy_dir, float* Fz_dir, float dt, int N){
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  // make sure i is in bounds
  if(i< N){
    //printf("%d pos called\n", i);
    // update velocities
    atomicAdd(&body[i*7+X_VEL], (Fx_dir[i]*dt)/body[i*7+MASS]);
    atomicAdd(&body[i*7+Y_VEL], (Fy_dir[i]*dt)/body[i*7+MASS]);
    atomicAdd(&body[i*7+Z_VEL], (Fz_dir[i]*dt)/body[i*7+MASS]);

	  // update positions
    atomicAdd(&body[i*7+X_POS], body[i*7+X_VEL]*dt);
    atomicAdd(&body[i*7+Y_POS], body[i*7+Y_VEL]*dt);
    atomicAdd(&body[i*7+Z_POS], body[i*7+Z_VEL]*dt);
  }
}
