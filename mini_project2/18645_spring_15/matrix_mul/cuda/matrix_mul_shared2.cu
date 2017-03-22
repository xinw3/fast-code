/*
    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix_mul.h"
#include "stdio.h"
#define TILE_WIDTH 32
#define BLOCK 32

namespace cuda
{
  __global__
  void
  matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension)
  {

    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    float sum = 0.0f;
    if(tx < sq_dimension && ty < sq_dimension){
      #pragma unroll
      for(int k = 0; k < sq_dimension; k++)
      {
        sum += sq_matrix_1[ty*sq_dimension + k] * sq_matrix_2[k*sq_dimension + tx];
        //__syncthreads();
      }
      sq_matrix_result[ty*sq_dimension + tx] = sum;
      __syncthreads();
    }

  }

  __global__
  void
  matrix_mul_kernel_shared(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension)
  {

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int result_row = blockRow * blockDim.x + threadRow;
    int result_col = blockCol * blockDim.x + threadCol;
    float sum = 0.0f;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    //printf("%d\n", sq_dimension / blockDim.x);
    for (int m = 0; m < (sq_dimension / blockDim.x); m++) {
        //printf("blockDim = %d \n", blockDim.x);
        // Row i of matrix A
        int i = m * blockDim.x + threadRow;

        // Column j of matrix A
        int j = m * blockDim.x + threadCol;

        // Load A(i,j) to shared mem
        //if (i < sq_dimension && result_row < sq_dimension) {
            As[threadRow][threadCol] = sq_matrix_1[result_row * sq_dimension + j];
            //printf("sq_matrix_1[%d * sq_dimension + %d] = %d \n", i, j, sq_matrix_1[i * sq_dimension + j]);
        //} else {
        //    As[threadRow][threadCol] = 0;
        //}
        // Load B(j,i) to shared mem
        //if (j < sq_dimension && result_col < sq_dimension) {
            Bs[threadRow][threadCol] = sq_matrix_2[i * sq_dimension + result_col]; // Global Mem Not coalesced
        //} else {
        //    Bs[threadCol][threadRow] = 0;
        //}
        // Synchronize before computation
        __syncthreads();
        // printf("As[%d][%d] = %d, ", threadRow, threadCol, As[threadRow][threadCol]);
        // printf("Bs[%d][%d] = %d \n", threadCol, threadRow, Bs[threadCol][threadRow]);
        // printf("sq_matrix_2[%d * sq_dimension + %d] = %d \n", j, i, sq_matrix_2[j * sq_dimension + i]);


        //if (result_col < TILE_WIDTH && result_col < TILE_WIDTH) {
        for (int k = 0; k < blockDim.x; k++) {
            // Accumulate for matrix C
            // printf("As[%d][%d] = %d * ", threadRow, k, As[threadRow][k]);
            // printf("Bs[%d][%d] = %d = ", k, Bs[k][threadCol]);
             sum += As[threadRow][k] * Bs[k][threadCol];
            //printf("sum = %d \n", sum);
        }
      //}
        // Synchronize
        __syncthreads();
    }

    //if (result_row < sq_dimension && result_col < sq_dimension) {
        sq_matrix_result[result_row * sq_dimension + result_col] = sum;
    //}

    // if(tx < sq_dimension && ty < sq_dimension){
    //   #pragma unroll
    //   for(int k = 0; k < sq_dimension; k++)
    //   {
    //     sum += sq_matrix_1[ty*sq_dimension + k] * sq_matrix_2[k*sq_dimension + tx];
    //     //__syncthreads();
    //   }
    //   sq_matrix_result[ty*sq_dimension + tx] = sum;
    //   __syncthreads();
    // }

  }

  void
  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension)
  {
    int size = sq_dimension * sq_dimension * sizeof(float);
    float *sq_matrix_1_d, *sq_matrix_2_d, *sq_matrix_result_d;

    /***************************************************
  1st Part: Allocation of memory on device memory
    ****************************************************/

    /* copy sq_matrix_1 and sq_matrix_2 to device memory */
    // sq_matrix_1 = (float*)malloc(size);
    cudaMalloc((void**) &sq_matrix_1_d, size);
    cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
    // sq_matrix_2 = (float*)malloc(size * sizeof(float));
    cudaMalloc((void**) &sq_matrix_2_d, size);
    cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);

    /*allocate sq_matrix_result on host */
    // sq_matrix_result = (float*)malloc(size * sizeof(float));
    cudaMalloc((void**) &sq_matrix_result_d, size);

    /***************************************************
   2nd Part: Inovke kernel
    ****************************************************/
    // // naive version
    // if (sq_dimension <= BLOCK) {
    //     dim3 dimBlock(sq_dimension, sq_dimension);
    //     dim3 dimGrid(1, 1);
    //     matrix_mul_kernel<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
    // } else {
    //     dim3 dimBlock(BLOCK, BLOCK);
    //     int blockNum = (sq_dimension * sq_dimension + BLOCK *BLOCK - 1)/ (BLOCK *BLOCK);
    //     dim3 dimGrid(blockNum,blockNum);
    //     matrix_mul_kernel<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
    // }
        // shared memory
        //printf("sq_dimension = %d\n", sq_dimension);
        if (sq_dimension <= TILE_WIDTH) {
            dim3 dimBlock(sq_dimension, sq_dimension);
            dim3 dimGrid(1, 1);

            // printf("blockDim = %d", blockDim.x);
            matrix_mul_kernel_shared<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
        } else {
            dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
            int blockNum = (sq_dimension * sq_dimension + TILE_WIDTH * TILE_WIDTH - 1)/ (TILE_WIDTH * TILE_WIDTH);
            printf("sq_dimension = %d, blockNum = %d", sq_dimension, blockNum);
            dim3 dimGrid(blockNum, blockNum);
            matrix_mul_kernel_shared<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
        }

    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    // int blockNum = ceil(sq_dimension*1.0/TILE_WIDTH);
    // dim3 dimGrid(blockNum,blockNum);


    /***************************************************
   3rd Part: Transfer result from device to host
    ****************************************************/
    cudaMemcpy(sq_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);
    cudaFree(sq_matrix_1_d);
    cudaFree(sq_matrix_2_d);
    cudaFree(sq_matrix_result_d);
    // free(sq_matrix_1);
  }
} // namespace cuda
