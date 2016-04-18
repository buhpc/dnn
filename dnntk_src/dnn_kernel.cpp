#include "dnn_kernel.h"
#include <math.h>
#include <immintrin.h>
#include <pthread.h>
#include <mpi.h>
#include <omp.h>

#define OMP_THREADS 10

#define PACK 8
typedef __m256 data_t;

extern "C" int setmatY(float *Y, float *B, int row, int col) 
{	
//#pragma offload target(mic) inout(data: length(row*col))
	data_t* a = (data_t*) Y;
	data_t* b = (data_t*) B;
	int idx;
	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col/PACK; j++)
		{
			idx = i*col + j;
			a[idx] = b[j];
		}
	}
	return 0;
}

extern "C" int sigmoidY(float *Y, int row, int col)
{
//#pragma offload target(mic) inout(data: length(row*col))
	data_t* a = (data_t*) Y;
	omp_set_num_threads(OMP_THREADS);
	data_t one = _mm256_set1_ps(1.0f);
	#pragma omp parallel for
	for(int i = 0;i < row*col/PACK; ++i)
	{
		a[i] = _mm256_div_ps(one,_m256_add_ps(one,_mm256_exp_ps(-a[i])));
	}
}

extern "C" int softmaxZ(float* in_vec, float* out_vec, int row, int col)
{
//#pragma offload target(mic) inout(data: length(row*col))
	data_t* a = (data_t*) in_vec;
	data_t* b = (data_t*) out_vec;
	data_t zero = _mm256_set1_ps(0.0f);
	data_t one = _mm256_set1_ps(1.0f);
	data_t tmp;
	data_t sumexp;
	int idx, base;
	float max;

	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		base = i*col + 0;
		max = in_vec[base];
		for(int j=1; j<col; j++)
		{
			idx = i*col + j;
			if(in_vec[idx] > max)
			{
				max = in_vec[idx];
			}
		}
		data_t sumexp = zero;
		data_t max_256 = _mm256_set1_ps(max);
		for(int j=0; j<col/PACK; j++)
		{
			idx = i*col + j;
			tmp = _mm256_exp_ps(_mm256_sub_ps(a[idx],max_256));
			sumexp = _mm256_add_ps(sumexp,tmp);
			a[idx] = tmp;
		}

		float last_exp = 0.0f;
		last_exp = last_exp & sumexp;
		last_exp = last_exp & (sumexp >> 32);
		last_exp = last_exp & (sumexp >> 64);
		last_exp = last_exp & (sumexp >> 96);
		last_exp = last_exp & (sumexp >> 128);	
		last_exp = last_exp & (sumexp >> 160);
		last_exp = last_exp & (sumexp >> 192);
		last_exp = last_exp & (sumexp >> 224);
		last_exp = last_exp & (sumexp >> 256);
		float last_tmp = 1.0f/last_exp;
		data_t last_we = _mm256_set1_ps(last_tmp);

		for(int j=0; j<col/PACK; j++)
		{
			idx = i*col + j;
			b[idx] = _mm256_mul_ps(a[idx],last_we);
		}
	}
}

extern "C" int errorTrans(float *E, float *Y, int row, int col)
{
//#pragma offload target(mic) inout(data: length(row*col))
	data_t* a = (data_t*) E;
	data_t* b = (data_t*) Y;
	data_t tmp;
	int idx;
	data_t one = _mm256_st1_ps(1);

	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col/PACK; j++)
		{
			idx = i*col + j;
			tmp = b[idx];
			a[idx] = _m256_mul_ps(a[idx],_m256_mul_ps(tmp,_mm256_sub_ps(1,tmp)));
		}
	}
	return 0;
}

extern "C" int errorOutput(float *E, float *Z, int *T, int row, int col)
{
//#pragma offload target(mic) inout(data: length(row*col))
	data_t* a = (data_t*) E;
	data_t* b = (data_t*) Z;
	data_t* c = (data_t*) T;
	int tmp;
	int idx;

	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			idx = i*col + j;	
			tmp = (j == c[i]) ? 1.0f:0.0f;
			E[idx] = Z[idx] - tmp;
		}
	}
	return 0;
}

extern "C" int updateW(float *W, float *Wdelta, int row, int col)
{
//#pragma offload target(mic) inout(data: length(row*col))
	data_t* a = (data_t*) W;
	data_t* b = (data_t*) Wdelta;
	int idx;

	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col/PACK; j++)
		{
			idx = i*col + j;
			a[idx] = _mm256_add_ps(a[idx],b[idx]);
		}
	}
	return 0;
}

extern "C" int updateB(float *E, float *B, float *Bdelta, int row, int col, float alpha)
{
//#pragma offload target(mic) inout(data: length(row*col))
	data_t* a = (data_t*) E;
	data_t* b = (data_t*) B;
	data_t* c = (data_t*) Bdelta;
	data_t zero = _mm256_set1_ps(0.0f);
	data_t alpha256 = _mm256_set1_ps(alpha);
	data_t sum;
	int idx;
	
	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<col/PACK; i++)
	{
		sum = zero;
		for(int j=0; j<row; j++)
		{
			idx = j*col + i;
			sum = _mm256_add_ps(sum,a[idx]);
		}
		
		c[i] = _mm256_add_ps(alpha256,sum);
		b[i] = _mm256_add_ps(b[i],c[i]);
	}
	return 0;
}

