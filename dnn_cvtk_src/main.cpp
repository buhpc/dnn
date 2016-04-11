#include "dnn_func.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

int main(int argc, char* argv[])
{
	puts(argv[0]);

	struct timeval timerStart, timerStop;
	gettimeofday(&timerStart, NULL);

	if((3 != argc )&&( 2 != argc) ){
		printf("ERROR: usage- <command> <config file> <iterator number>\n");
		exit(1);
	}
	const char* initNum = NULL;
	if(2 == argc)
		initNum = "0";
	else
		initNum = argv[2];
	CpuArg cpuArg = {0};
	ChunkContainer oneChunk = {0};
	NodeArg nodeArg = {0};

	int ret = GetInitFileConfig(argv[1],initNum,cpuArg);
	if(0 != ret) {
		printf("Error happens: GetInitFileCofig\n");
		return ret;
	}

	InitNodeConfig(cpuArg,nodeArg);
	
	//cross valid
	int validSamples = 0;
	int totalSamples = 0;
	int chunkSize    = cpuArg.chunkSize;

	int *h_R = nodeArg.d_R;
	int *h_T = nodeArg.d_T;

	fprintf(cpuArg.pLogFile,"starting cross validing:\n");
	fflush(cpuArg.pLogFile);

	while(FetchOneChunk(cpuArg, oneChunk))
	{
		while (FetchOneBunch(oneChunk, nodeArg)) {
			dnnClassify(nodeArg);

			for (int i = 0; i < nodeArg.numN; i++) {
				if (h_R[i] == h_T[i]) 
					validSamples++;
			}
			totalSamples += nodeArg.numN;
		}
	}

	
	fprintf(cpuArg.pLogFile,"cv over\n");
	fflush(cpuArg.pLogFile);
	fprintf(cpuArg.pLogFile,"total samples:%d \ncorrect samples: %d\naccuracy: %f\n", totalSamples,\
			validSamples, (float)validSamples / (float)totalSamples * 100.f);
	fflush(cpuArg.pLogFile);
	WriteWts(nodeArg, cpuArg);
	gettimeofday(&timerStop, NULL);
	float timerElapsed = 1000.0 * (timerStop.tv_sec - timerStart.tv_sec) + (timerStop.tv_usec - timerStart.tv_usec) / 1000.0;
	fprintf(cpuArg.pLogFile,"total time cost: %fms\n", timerElapsed);
	fflush(cpuArg.pLogFile);

	UninitProgramConfig(cpuArg,nodeArg,oneChunk);

	return 0;
}
