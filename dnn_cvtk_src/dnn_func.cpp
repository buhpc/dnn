#include "dnn_func.h"
#include "dnn_kernel.h"

extern "C" int dnnClassify(NodeArg &nodeArg)
{
	float  *d_X = nodeArg.d_X;
	int    *d_R = nodeArg.d_R;
	float **d_Y = nodeArg.d_Y;
	float **d_W = nodeArg.d_W;
	float **d_B = nodeArg.d_B;

	float one = 1.0f;
	float zero  = 0.0f;

	int numN = nodeArg.numN; //size of minibatch
	int numL = nodeArg.dnnLayerNum - 1; //layer nums
	int numD = nodeArg.dnnLayerArr[0]; //node nums of input layer
	int*numA = &(nodeArg.dnnLayerArr[1]); //node nums of hiden layer and output layer

	_Cilk_for(int i = 0; i < numL; i++) {
		setmatY(d_Y[i], d_B[i], numN, numA[i]);
		
		if (i == 0){ 
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, numN, numA[i], numD, \
						one, d_X, numD, d_W[i], numA[i], one, d_Y[i], numA[i]);
		} else {
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, numN, numA[i], numA[i-1], \
						one, d_Y[i-1], numA[i-1], d_W[i], numA[i],  one, d_Y[i], numA[i]);
		}
		
		if (i == numL-1) { //softmax on output layer
			softmaxZ(d_Y[i], d_Y[i], numN, numA[i]);
			reduceIndex(d_Y[i], d_R, numN, numA[i]);
		} else { //sigmod on hiden layers
			sigmoidY(d_Y[i], numN, numA[i]);
		}
	}

	return 0;

}

