#ifndef DNN_KERNEL_H
#define DNN_KERNEL_H

/******************************************
**kernel function used in forward process*
******************************************/
extern "C" int setmatY (float *Y, float *B, int row, int col);//expand vector B to matrix Y
extern "C" int sigmoidY(float *Y, int row, int col);//handler 0~NUM_LAYER-1 layer
extern "C" int softmaxZ(float* inZ, float* outZ, int row, int col);//handle output layer

/******************************************
**kernel function used in classify process*
******************************************/
extern "C" int reduceIndex(float *Y, int *R, int row, int col);//R[i] = max(Y[i][j]) where 0<=j<=col;

#endif
