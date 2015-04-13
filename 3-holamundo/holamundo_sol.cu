#include <stdio.h>

#include <cuda.h>
#include <stdio.h>
//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"

__global__ void kernel(char * m){

   int i = threadIdx.x;
   //cuPrintf("Thread %d: %c \n",i,m[i]);
   printf("Thread %d: %s \n",i,(const char *)m);
   return;
}

int main (int argc, char ** argv) {

   //intitialize cuPrintf
   //cudaPrintfInit();

   int tamM = 16;
   char * m  = (char *) malloc(tamM);
   char * dm = NULL;

   _snprintf_s(m,8,8, "Hello!!\0");

   fprintf(stderr,"Al hacer la reserva: %s \n",cudaGetErrorString(cudaMalloc(&dm,tamM)));

   fprintf(stderr,"Al hacer la copia:   %s \n",cudaGetErrorString(cudaMemcpy(dm,m,tamM,cudaMemcpyHostToDevice)));

   fprintf(stderr,"Soy el HOST voy a mandar a dispositivo: %s\n",m);

   kernel<<<1,tamM>>>(dm);

   fprintf(stderr,"Mensaje: %s\n",m);

   //display the device's greetings
   //cudaPrintfDisplay();

   //clean up
   //cudaPrintfEnd();

   cudaFree(dm);
   
   return 0;
}