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
#define TILE_WIDTH 2
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
    if (sq_dimension <= BLOCK) {
        dim3 dimBlock(sq_dimension, sq_dimension);
        dim3 dimGrid(1, 1);
        matrix_mul_kernel<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
    } else {
        dim3 dimBlock(BLOCK, BLOCK);
        int blockNum = (sq_dimension * sq_dimension + BLOCK *BLOCK - 1)/ (BLOCK *BLOCK);
        dim3 dimGrid(blockNum,blockNum);
        matrix_mul_kernel<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
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
