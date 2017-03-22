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

    team034: Xin Wang, Xinran Fang
*/

#include <omp.h>
//#include <iostream>
#include "matrix_mul.h"
#include <xmmintrin.h>
#include <string.h>
#include <x86intrin.h>

namespace omp
{
    void Transpose(unsigned int sq_dimension, float *sq_matrix_2, float *sq_matrix_2_transpose)
    {

        unsigned int j = 0;
        for(unsigned int i = 0; i < sq_dimension; i++)
        {
            #pragma unroll(4)
            for(j = 0 ; j < sq_dimension; j++)
            {
                // loop unrolling.
                sq_matrix_2_transpose[j * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j];
                // sq_matrix_2_transpose[(j + 1) * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j + 1];
                // sq_matrix_2_transpose[(j + 2) * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j + 2];
                // sq_matrix_2_transpose[(j + 3) * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j + 3];
                // sq_matrix_2_transpose[(j + 4) * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j + 4];
                // sq_matrix_2_transpose[(j + 5) * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j + 5];
                // sq_matrix_2_transpose[(j + 6) * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j + 6];
                // sq_matrix_2_transpose[(j + 7) * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j + 7];
            }
            for(; j < sq_dimension; j++)
            {
                sq_matrix_2_transpose[j * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j];
            }
        }
    }
    void
    matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, 
                        float *sq_matrix_result, unsigned int sq_dimension)
    {
        float *sq_matrix_2_transpose;
        sq_matrix_2_transpose = (float*) malloc (sizeof(float) * sq_dimension * sq_dimension);
        memset(sq_matrix_2_transpose, 0, sizeof(float) * sq_dimension * sq_dimension);

        Transpose(sq_dimension, sq_matrix_2, sq_matrix_2_transpose);

        #pragma omp parallel num_threads(4)
        for (unsigned int i = 0; i < sq_dimension; i++) 
        {
              for(unsigned int j = 0; j < sq_dimension; j++) 
	          {         	    
                __m128 c = _mm_setzero_ps();
          	    unsigned int k;
                for ( k = 0; k < sq_dimension - 4; k += 4) 
                {
        	        __m128 first = _mm_loadu_ps(sq_matrix_1 + i * sq_dimension + k);
                    __m128 second = _mm_loadu_ps(sq_matrix_2_transpose + j * sq_dimension + k);
                    c = _mm_add_ps(c,_mm_mul_ps(first, second));
                }
                c = _mm_hadd_ps(c, c);
                c = _mm_hadd_ps(c, c);
                _mm_store_ss((sq_matrix_result+i * sq_dimension + j),c);
      
                for(;k < sq_dimension; k++)
                {
                    sq_matrix_result[i * sq_dimension + j] += 
                        sq_matrix_1[i * sq_dimension + k] * sq_matrix_2_transpose[j * sq_dimension + k];
                }
            } 
        }// End of parallel region
    }
} //namespace omp
