#include "dnn_kernel.h"
#include <math.h>

extern "C" int setmatY(float *Y, float *B, int row, int col) 
{
	int idx;
	_Cilk_for(int i=0; i<row; i++)
	{
		_Cilk_for(int j=0; j<col; j++)
		{
			idx = i*col + j;
			Y[idx] = B[j];
		}
	}
	return 0;
}

extern "C" int sigmoidY(float *Y, int row, int col)
{
	_Cilk_for(int i = 0;i < row*col; ++i)
	{
		Y[i] = 1.0f/(1.0f + expf(-Y[i]));
	}
}

extern "C" int softmaxZ(float* in_vec, float* out_vec, int row, int col)
{
	int idx, base;
	float max, tmp;
	float sumexp = 0.0f;
	_Cilk_for(int i=0; i<row; i++)
	{
		base = i*col + 0;
		max = in_vec[base];
		_Cilk_for(int j=1; j<col; j++)
		{
			idx = i*col + j;
			if(in_vec[idx] > max)
			{
				max = in_vec[idx];
			}
		}
		sumexp = 0.0f;
		_Cilk_for(int j=0; j<col; j++)
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

extern "C" int reduceIndex(float *Y, int *R, int row, int col)
{
	int idx;
	int base;
	int maxidx;
	for(int i=0; i<row; i++)
	{
		base = i * col + 0;//Y[i, 0]
		maxidx = 0;
		for(int j=1; j<col; j++)
		{
			idx = i*col + j;//Y[i, j]
			if(Y[base] < Y[idx])
			{
				base = idx;
				maxidx = j;
			}
		}
		R[i] = maxidx;
	}
	return 0;
}
