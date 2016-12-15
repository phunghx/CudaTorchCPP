#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/cudaconv2.cuh"
#define DSIZE 1024
#define DVAL 20
#define nTPB 256
//#include <cuda_runtime.h>

__global__ void phglobal_reduce_kernel(float * d_out, float * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

__global__ void phshmem_reduce_kernel(float * d_out, const float * d_in)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void my_kernel2(int *data){
  int idx = threadIdx.x + (blockDim.x *blockIdx.x);
  if (idx < DSIZE) data[idx] =+ DVAL;
}

int my_test_func2(){

  int *d_data, *h_data;
  h_data = (int *) malloc(DSIZE * sizeof(int));
  if (h_data == 0) {printf("malloc fail\n"); exit(1);}
  cudaMalloc((void **)&d_data, DSIZE * sizeof(int));
  cudaCheckErrors("cudaMalloc fail");
  for (int i = 0; i < DSIZE; i++) h_data[i] = 0;
  cudaMemcpy(d_data, h_data, DSIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy fail");
  my_kernel2<<<((DSIZE+nTPB-1)/nTPB), nTPB>>>(d_data);
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel");
  cudaMemcpy(h_data, d_data, DSIZE * sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy 2");
  for (int i = 0; i < DSIZE; i++)
    if (h_data[i] != DVAL) {printf("Results check failed at offset %d, data was: %d, should be %d\n", i, h_data[i], DVAL); exit(1);}
  printf("Results check 2 passed!\n");
  return 0;
}

void phungReduce(float * d_out, float * d_intermediate, float * d_in, 
            int size, bool usesSharedMemory)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;
    if (usesSharedMemory)
    {
        phshmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
            (d_intermediate, d_in);
    }
    else
    {
        phglobal_reduce_kernel<<<blocks, threads>>>
            (d_intermediate, d_in);
    }
    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    if (usesSharedMemory)
    {
        phshmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
            (d_out, d_intermediate);
    }
    else
    {
        phglobal_reduce_kernel<<<blocks, threads>>>
            (d_out, d_intermediate);
    }
}


