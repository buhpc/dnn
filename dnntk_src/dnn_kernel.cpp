#include "dnn_kernel.h"
#include <math.h>
#include <immintrin.h>
#include <pthread.h>
#include <mpi.h>
#include <omp.h>

#define OMP_THREADS 10

#define PACK 8
typedef __m256 data_t;


int rank;//MPI Global Variable

extern "C" int setmatY(float *Y, float *B, int row, int col) 
{	
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
	//data_t* a = (data_t*) Y;
	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i = 0;i < row*col/PACK; ++i)
	{
		Y[i] = 1.0f/(1.0f + expf(-Y[i]));
	}
}

extern "C" int softmaxZ(float* in_vec, float* out_vec, int row, int col)
{
	data_t* a = (data_t*) in_vec;
	data_t* b = (data_t*) out_vec;
	int idx, base;
	float max, tmp;
	float sumexp = 0.0f;
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
		sumexp = 0.0f;
		for(int j=0; j<col; j++)
		{
			idx = i*col + j;
			tmp = expf(in_vec[idx] - max);
			sumexp += tmp;
			in_vec[idx] = tmp;
		}
		tmp = 1.0f/sumexp;
		for(int j=0; j<col; j++)
		{
			idx = i*col + j;
			out_vec[idx] = in_vec[idx] * tmp;
		}
	}
}

extern "C" int errorTrans(float *E, float *Y, int row, int col)
{
	data_t* a = (data_t*) E;
	data_t* b = (data_t*) Y;
	float tmp;
	int idx;
	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col/PACK; j++)
		{
			idx = i*col + j;
			tmp = b[idx];
			a[idx] = a[idx] * tmp * (1-tmp);
		}
	}
	return 0;
}

extern "C" int errorOutput(float *E, float *Z, int *T, int row, int col)
{
	data_t* a = (data_t*) E;
	data_t* b = (data_t*) Z;
	data_t* c = (data_t*) T;
	float tmp;
	int idx;
	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		//Make sure to pack j into vector to compare to c
		for(int j=0; j<col/PACK; j++)
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
	int idx;
	data_t* a = (data_t*) W;
	data_t* b = (data_t*) Wdelta;
	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col/PACK; j++)
		{
			idx = i*col + j;
			a[idx] += b[idx];
		}
	}
	return 0;
}

extern "C" int updateB(float *E, float *B, float *Bdelta, int row, int col, float alpha)
{	
	int idx;
	data_t* a = (data_t*) E;
	data_t* b = (data_t*) B;
	data_t* c = (data_t*) Bdelta;
	float sum = 0.0f;
	omp_set_num_threads(OMP_THREADS);
	#pragma omp parallel for
	for(int i=0; i<col/PACK; i++)
	{
		sum = 0.0;
		for(int j=0; j<row; j++)
		{
			idx = j*col + i;
			sum += a[idx];
		}
		
		c[i] = alpha * sum;
		b[i] += c[i];
	}
	return 0;
}

