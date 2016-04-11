#include "dnn_func.h"
#include "dnn_kernel.h"

extern "C" int dnnForward(NodeArg &nodeArg)
{
	float  *d_X = nodeArg.d_X;
	float **d_Y = nodeArg.d_Y;
	float **d_W = nodeArg.d_W;
	float **d_B = nodeArg.d_B;

	float one = 1.0f;
	float zero  = 0.0f;

	int numN = nodeArg.numN; //size of minibatch
	int numL = nodeArg.dnnLayerNum - 1; //layer nums
	int numD = nodeArg.dnnLayerArr[0]; //node nums of input layer
	int*numA = &(nodeArg.dnnLayerArr[1]); //node nums of hiden layer and output layer

	for (int i = 0; i < numL; i++) {
		setmatY(d_Y[i], d_B[i], numN, numA[i]);
		
		if (i == 0){ 
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, numN, numA[i], numD, \
						one, d_X, numD, d_W[i],  numA[i], one, d_Y[i], numA[i]);
		} else {
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, numN, numA[i], numA[i-1], \
						one, d_Y[i-1], numA[i-1], d_W[i], numA[i], one, d_Y[i], numA[i]);
		}

		if (i == numL-1) { //softmax on output layer
			softmaxZ(d_Y[i], d_Y[i], numN, numA[i]);
		} else { //sigmod on hiden layers
			sigmoidY(d_Y[i], numN, numA[i]);
		}
	}

	return 0;
}

extern "C" int dnnBackward(NodeArg &nodeArg)
{
	float **d_Y = nodeArg.d_Y;
	float **d_W = nodeArg.d_W;
	float **d_E = nodeArg.d_E;
	int    *d_T = nodeArg.d_T;

	float one = 1.0f;
	float zero  = 0.0f;

	int numN = nodeArg.numN; //size of minibatch
	int numL = nodeArg.dnnLayerNum - 1; //layer nums
	int numD = nodeArg.dnnLayerArr[0]; //node nums of input layer
	int*numA = &(nodeArg.dnnLayerArr[1]); //node nums of hiden layer and output layer

	errorOutput(d_E[numL-1], d_Y[numL-1], d_T, numN, numA[numL-1]);

	for (int i = numL-2; i>=0; i--) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, numN, numA[i], numA[i+1], \
					one, d_E[i+1], numA[i+1], d_W[i+1], numA[i+1], zero, d_E[i], numA[i]);
		errorTrans(d_E[i], d_Y[i], numN, numA[i]);
	}

	return 0;
}

extern "C" int dnnUpdate(NodeArg &nodeArg)
{
	float  *d_X    = nodeArg.d_X;
	float **d_Y    = nodeArg.d_Y;
	float **d_E    = nodeArg.d_E;
	float **d_W    = nodeArg.d_W;
	float **d_B    = nodeArg.d_B;
	float **d_Wdta = nodeArg.d_Wdelta;
	float **d_Bdta = nodeArg.d_Bdelta;

	float lrate = nodeArg.lRate;

	int numN = nodeArg.numN; //size of minibatch
	int numL = nodeArg.dnnLayerNum - 1; //layer nums
	int numD = nodeArg.dnnLayerArr[0]; //node nums of input layer
	int*numA = &(nodeArg.dnnLayerArr[1]); //node nums of hiden layer and output layer

	//alpha: curlate, beta: power
	float alpha = -lrate / numN;
        float zero = 0.0f;
	float one = 1.0f;

	//update weight and bias of input layer
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, numD, numA[0], numN, \
				alpha, d_X, numD, d_E[0], numA[0], zero, d_Wdta[0], numA[0]);
	updateW(d_W[0], d_Wdta[0], numD, numA[0]);
	updateB(d_E[0], d_B[0], d_Bdta[0], numN, numA[0], alpha);

	//update weight and bias of hiden layers
	for (int i = 1; i < numL; i++) {
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, numA[i-1], numA[i],  numN, \
					alpha, d_Y[i-1], numA[i-1], d_E[i], numA[i], zero, d_Wdta[i], numA[i]);
		//update
		updateW(d_W[i], d_Wdta[i], numA[i-1], numA[i]);
		updateB(d_E[i], d_B[i], d_Bdta[i], numN, numA[i], alpha);
	}

	return 0;
}

