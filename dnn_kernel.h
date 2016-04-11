#ifndef DNN_KERNEL_H
#define DNN_KERNEL_H

/******************************************
**kernel function used in forward process*
******************************************/
extern "C" int setmatY (float *Y, float *B, int row, int col);//expand vector B to matrix Y
extern "C" int sigmoidY(float *Y, int row, int col);
extern "C" int softmaxZ(float* inZ, float* outZ, int row, int col);//handle output layer

/******************************************
**kernel function used in backward process*
******************************************/
extern "C" int errorTrans (float *E, float *Y, int row, int col);
extern "C" int errorOutput(float *E, float *Z, int *T, int row, int col);//caculate output layer's error

/******************************************
**kernel function used in update process*
******************************************/
extern "C" int updateW(float *W, float *Wdelta, int row, int col);//W+=Wdelta
extern "C" int updateB(float *E, float *B, float *Bdelta, int row, int col, float alpha);

#endif
