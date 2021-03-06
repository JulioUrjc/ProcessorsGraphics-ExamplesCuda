#include <stdio.h>
#include "timer.h"

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {

	int bx = blockIdx.x;  
	int by = blockIdx.y;  
	int tx = threadIdx.x;  
	int ty = threadIdx.y;
	
	int r = by * blockDim.y + ty;	
	int c = bx * blockDim.x + tx;
	int dimC = numAColumns;
	
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];	

	float value = 0.0;
	
	for (int i=0; i < (dimC-1)/TILE_WIDTH +1; i++){


		if ((r < numCRows) && ((i*TILE_WIDTH + tx)< dimC)){
			ds_A[ty][tx]=A[r*dimC + i*TILE_WIDTH + tx];
		} else {
			ds_A[ty][tx]=0.0;
		}
		
		if ((c < numCColumns) && ((i*TILE_WIDTH + ty)< dimC)){
			ds_B[ty][tx]=B[(i*TILE_WIDTH + ty)*numBColumns + c];
		} else {
			ds_B[ty][tx]=0.0;
		}
		
		__syncthreads();
		
		for (int j=0; j<TILE_WIDTH; j++){
			value += ds_A[ty][j] * ds_B[j][tx];	
		}

		__syncthreads();
		
	}
	
	if ((r < numCRows) && (c < numCColumns)){
		C[r*numCColumns+c] = value;
	}
}


void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

int main(int argc, char ** argv) {
	GpuTimer timer;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

	// Conseguir matrices de entrada. Random
	if (argc != 5){
		fprintf(stderr,"%s numrowsA numcolumnsA numrowsB numcolumnsB\n", argv[0]);
		return 1;
	}

	numARows = atoi(argv[1]);
	numAColumns = atoi(argv[2]);
	numBRows = atoi(argv[3]);
	numBColumns = atoi(argv[4]);
    //@@ Set numCRows and numCColumns
	if (numAColumns ==  numBRows){
		numCRows = numARows;
		numCColumns = numBColumns;
	} else {
	    fprintf(stderr, "The multiplication can not be made because %d columns of matrix A is not equal to %d rows of matrix B\n", numAColumns, numBRows);
		return 1;
	}

	// Initialize host memory
	cudaHostAlloc((void **) &hostA, numARows*numAColumns*sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void **) &hostB, numBRows*numBColumns*sizeof(float), cudaHostAllocDefault);
	const float valB = 0.01f;
	constantInit(hostA, numARows*numAColumns, 1.0f);
        constantInit(hostB, numBRows*numBColumns, valB);

    //@@ Allocate the hostC matrix
	cudaHostAlloc((void **) &hostC, numCRows*numCColumns*sizeof(float), cudaHostAllocDefault);

    //@@ Allocate GPU memory here
    cudaMalloc((void **) &deviceA,numARows*numAColumns*sizeof(float));
    cudaMalloc((void **) &deviceB,numBRows*numBColumns*sizeof(float));
    cudaMalloc((void **) &deviceC,numCRows*numCColumns*sizeof(float));

    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA,hostA,numARows*numAColumns*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB,hostB,numBRows*numBColumns*sizeof(float),cudaMemcpyHostToDevice);	
   
    //@@ Initialize the grid and block dimensions here
	dim3 gridSize((numCColumns-1)/TILE_WIDTH +1, (numCRows-1)/TILE_WIDTH +1, 1);
	dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);

    timer.Start();
    //@@ Launch the GPU Kernel here
	matrixMultiplyShared<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns,
                   numCRows, numCColumns);
    cudaDeviceSynchronize();
    timer.Stop();
    
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC,numCRows*numCColumns*sizeof(float),cudaMemcpyDeviceToHost);	

    //@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	
    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula 
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps 
    double eps = 1.e-6 ; // machine zero
    for (int i = 0; i < (int)(numCRows * numCColumns); i++)
    {
        double abs_err = fabs(hostC[i] - (numAColumns * valB));
        double dot_length = numAColumns;
        double abs_val = fabs(hostC[i]);
        double rel_err = abs_err/abs_val/dot_length ;
        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, hostC[i], numAColumns*valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);

    return 0;
}

