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

#include <omp.h>
#include <iostream>
#include "matrix_mul.h"
#include <xmmintrin.h>
#include <string.h>
#include <x86intrin.h>
//#include <stdlib.h>
#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
namespace omp
{
    // void trans_sse(unsigned int sq_dimension, float *sq_matrix_2, float *sq_matrix_2_transpose, 
    //                 unsigned int i, unsigned int j)
    // {
    //     __m128 row0 = _mm_loadu_ps(sq_matrix_2 + i * sq_dimension + j);
    //     __m128 row1 = _mm_loadu_ps(sq_matrix_2 + (i + 1) * sq_dimension + j);
    //     __m128 row2 = _mm_loadu_ps(sq_matrix_2 + (i + 2) * sq_dimension + j);
    //     __m128 row3 = _mm_loadu_ps(sq_matrix_2 + (i + 3) * sq_dimension + j);
    //     _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    //     _mm_storeu_ps((sq_matrix_2_transpose + i * sq_dimension + j), row0);
    //     _mm_storeu_ps((sq_matrix_2_transpose + (i + 1) * sq_dimension + j), row1);
    //     _mm_storeu_ps((sq_matrix_2_transpose + (i + 2) * sq_dimension + j), row2);
    //     _mm_storeu_ps((sq_matrix_2_transpose + (i + 3) * sq_dimension + j), row3);

    //     // __m128 row0 = _mm_loadu_ps(sq_matrix_2 + 0 * ld);
    //     // __m128 row1 = _mm_loadu_ps(sq_matrix_2 + 1 * ld);
    //     // __m128 row2 = _mm_loadu_ps(sq_matrix_2 + 2 * ld);
    //     // __m128 row3 = _mm_loadu_ps(sq_matrix_2 + 3 * ld);
    //     // _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    //     // _mm_storeu_ps((sq_matrix_2_transpose + 0 * ld), row0);
    //     // _mm_storeu_ps((sq_matrix_2_transpose + 1 * ld), row1);
    //     // _mm_storeu_ps((sq_matrix_2_transpose + 2 * ld), row2);
    //     // _mm_storeu_ps((sq_matrix_2_transpose + 3 * ld), row3);
    // }
    void Transpose(unsigned int sq_dimension, float *sq_matrix_2, float *sq_matrix_2_transpose)
    {


        unsigned int i = 0, j = 0;
        //#pragma unroll(4)
        for(i = 0; i <= sq_dimension - 4; i += 1)
        {
            #pragma unroll(4)
            
            for(j = 0 ; j <= sq_dimension - 4; j += 4)
            {
                //sq_matrix_2_transpose[j * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j];
                __m128 row0 = _mm_loadu_ps(sq_matrix_2 + i * sq_dimension + j);
                __m128 row1 = _mm_loadu_ps(sq_matrix_2 + (i + 1) * sq_dimension + j);
                __m128 row2 = _mm_loadu_ps(sq_matrix_2 + (i + 2) * sq_dimension + j);
                __m128 row3 = _mm_loadu_ps(sq_matrix_2 + (i + 3) * sq_dimension + j);
                _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
                if (j == 0)
                {
                    _mm_storeu_ps((sq_matrix_2_transpose + i * sq_dimension + j), row0);
                    _mm_storeu_ps((sq_matrix_2_transpose + (i + 1) * sq_dimension + j), row1);
                    _mm_storeu_ps((sq_matrix_2_transpose + (i + 2) * sq_dimension + j), row2);
                    _mm_storeu_ps((sq_matrix_2_transpose + (i + 3) * sq_dimension + j), row3);
                } else {
                    _mm_storeu_ps((sq_matrix_2_transpose + (i + j) * sq_dimension + j - 4 * i), row0);
                    _mm_storeu_ps((sq_matrix_2_transpose + (i + j + 1) * sq_dimension + j - 4 * i), row1);
                    _mm_storeu_ps((sq_matrix_2_transpose + (i + j + 2) * sq_dimension + j - 4 * i), row2);
                    _mm_storeu_ps((sq_matrix_2_transpose + (i + j + 3) * sq_dimension + j - 4 * i), row3);
                }
                //std::cout<<"hhhhhh"<<std::endl;
            }
            
            for (; j < sq_dimension; j++) 
            {
                //std::cout<<"why segfault"<<std::endl;
                sq_matrix_2_transpose[j * sq_dimension + i] = sq_matrix_2[i * sq_dimension + j];
            }
            
        }
        for (i = 0; i < sq_dimension; i++)
            {
                for (j = 4; j < sq_dimension; j += 4) 
                {
                //std::cout<<"why segfault"<<std::endl;
                    sq_matrix_2_transpose[j * sq_dimension + i] = sq_matrix_2_transpose[i * sq_dimension + j];
                }
            }
        //unsigned int k = j;
        
    }
    void
    matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, 
            float *sq_matrix_result, unsigned int sq_dimension)
    {
        float *sq_matrix_2_transpose;
        sq_matrix_2_transpose = (float*) malloc (sizeof(float) * sq_dimension * sq_dimension);
        memset(sq_matrix_2_transpose, 0, sizeof(float) * sq_dimension * sq_dimension);

        std::cout << "------------ORIGINAL RESULT----------" << std::endl;
        for (unsigned int i = 0; i < sq_dimension; i++) {
            for (unsigned int j = 0; j < sq_dimension; j++) {
                std::cout << sq_matrix_2[i * sq_dimension +j] << " ";
            }
            std::cout << " "<<std::endl;
        }
        Transpose(sq_dimension, sq_matrix_2, sq_matrix_2_transpose);
        //omp_set_num_threads(4);

        std::cout << "------------ Transpose ----------" << std::endl;
        for (unsigned int i = 0; i < sq_dimension; i++) {
            for (unsigned int j = 0; j < sq_dimension; j++) {
                std::cout << sq_matrix_2_transpose[i * sq_dimension +j] << " ";
            }
            std::cout << " "<<std::endl;
        }

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
