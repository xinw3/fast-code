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
namespace omp
{
    inline void Transpose(unsigned int size, float *matrix){
        for(unsigned int i = 0; i < size; i++){
            for(unsigned int j = 0 ; j < size; j++){
                std::swap(matrix[i*size+j], matrix[j*size+i]);
            }
        }
    }
    inline void transpose_4x4()
    void
    matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, 
            float *sq_matrix_result, unsigned int sq_dimension )
    {
        float *sq_matrix_2_transpose;

        Transpose(sq_dimension, sq_matrix_2);
        omp_set_num_threads(4);
        #pragma omp parallel for
        for (unsigned int i = 0; i < sq_dimension; i++) 
        {
	          for(unsigned int j = 0; j < sq_dimension; j++) 
	          {       
          	    __m128 c = _mm_setzero_ps();
          	    unsigned int k; 
                for ( k = 0; k < sq_dimension - 4; k = k + 4) 
                {
        	        __m128 first = _mm_loadu_ps(&sq_matrix_1[i * sq_dimension + k]);
                  __m128 second = _mm_loadu_ps(&sq_matrix_2[j * sq_dimension + k]);
                  c = _mm_add_ps(c,_mm_mul_ps(first, second));
                  //sq_matrix_1[i*sq_dimension + k] * sq_matrix_2[k*sq_dimension + j];
                }
                _mm_store_ss(&sq_matrix_result[i * sq_dimension + j],c);
                __m128 c_left=_mm_setzero_ps();
                __m128 m1_left=_mm_setzero_ps();
                __m128 m2_left=_mm_setzero_ps();
                for(;k < sq_dimension; k++)
                {
                    m1_left = _mm_loadu_ps(&sq_matrix_1[i * sq_dimension + k]);
                    m2_left = _mm_loadu_ps(&sq_matrix_2[k * sq_dimension + j]);
                    c_left = _mm_mul_ps(m1_left, m2_left);
                }
                _mm_store_ps(&sq_matrix_result[i * sq_dimension + j], c_left);
        
                //sq_matrix_result[i*sq_dimension + j] = c;
            } 
        }// End of parallel region
    }
  
} //namespace omp
