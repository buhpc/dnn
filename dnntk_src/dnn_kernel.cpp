#include "dnn_kernel.h"z
#include <math.h>
#include <immintrin.h>

extern "C" int setmatY(float *Y, float *B, int row, int col) 
{
	#pragma omp parallel for
	int idx;
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			idx = i*col + j;
			Y[idx] = B[j];
		}
	}
	return 0;
}

extern "C" int sigmoidY(float *Y, int row, int col)
{
	#pragma omp parallel for
	for(int i = 0;i < row*col; ++i)
	{
		Y[i] = 1.0f/(1.0f + expf(-Y[i]));
	}
}

extern "C" int softmaxZ(float* in_vec, float* out_vec, int row, int col)
{
	int idx, base;
	float max, tmp;
	float sumexp = 0.0f;
	#pragma omp paralleel for
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
	float tmp;
	int idx;
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			idx = i*col + j;
			tmp = Y[idx];
			E[idx] = E[idx] * tmp * (1-tmp);
		}
	}
	return 0;
}

extern "C" int errorOutput(float *E, float *Z, int *T, int row, int col)
{
	float tmp;
	int idx;
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			idx = i*col + j;
			tmp = (j == T[i]) ? 1.0f:0.0f;
			E[idx] = Z[idx] - tmp;
		}
	}
	return 0;
}

extern "C" int updateW(float *W, float *Wdelta, int row, int col)
{
	int idx;
	#pragma omp parallel for
	for(int i=0; i<row; i++)
	{
		for(int j=0; j<col; j++)
		{
			idx = i*col + j;
			W[idx] += Wdelta[idx];
		}
	}
	return 0;
}

extern "C" int updateB(float *E, float *B, float *Bdelta, int row, int col, float alpha)
{	
	int idx;
	__m256* Src1 = (__m256*) E;
	__m256* Src2 = (__m256*) B;
	__m256* Src3 = (__m256*) Bdelta;
	//float sum = 0.0f;
	__m256* sum;
	for(int i=0; i<col/8; i++)
	{
		sum = (__m256)0;
		for(int j=0; j<row/8; j++)
		{
			idx = j*col + i;
			sum += Src1[idx];
		}
		
		Src3[i] = (__m256)alpha * sum;
		Src2[i] += Src3[i];
	}
	return 0;
}

