#include <stdio.h>

#include <cuda.h>
#include <stdio.h>

#define NUM_THREADS 500000
#define BLOCK_WIDTH 500

__global__ void kernel(char * m){

	printf("%s soy el thread %d \n",m, threadIdx.x);
   return;
}

int main (int argc, char ** argv) {


   int tamM = 8;
   char * m  = (char *) malloc(tamM);
   char * dm = NULL;

   _snprintf_s(m,8,8, "Hello!!\0");

   fprintf(stderr,"Al hacer la reserva: %s \n",cudaGetErrorString(cudaMalloc(&dm,tamM)));
   fprintf(stderr,"Al hacer la copia:   %s \n",cudaGetErrorString(cudaMemcpy(dm,m,tamM,cudaMemcpyHostToDevice)));
   fprintf(stderr,"Soy el HOST voy a mandar a dispositivo: %s\n",m);

   //Launch kernel
   kernel<<< 1, 16 >>>(dm);

   int dev_count;
   cudaDeviceProp prop;
   cudaGetDeviceCount(&dev_count);
   for (int i = 0; i < dev_count; i++){
	   cudaGetDeviceProperties(&prop, i);
	   //fprintf(stderr, " %s\n", prop);
   }

   fprintf(stderr,"Mensaje: %s\n",m);

   cudaFree(dm);
   
   return 0;
}
