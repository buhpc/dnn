#include "dnn_func.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

int main(int argc, char* argv[])
{
	int rank;
	int size;
	MPI_init(&argc,&argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
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
	
	//training process
	fprintf(cpuArg.pLogFile,"start training:\n");
	fflush(cpuArg.pLogFile);
	int  chunkCnt  = 0;
	int  readSize  = 0;
	int  countCnt = 0;
	while ((readSize = FetchOneChunk(cpuArg, oneChunk)) && readSize) {
		fprintf(cpuArg.pLogFile,"--chunk(%d) : containing samples %d\n", chunkCnt++, readSize);
		fflush(cpuArg.pLogFile);
		while (FetchOneBunch(oneChunk, nodeArg) == cpuArg.bunchSize) {
			printf("train: %d \n", ++countCnt);
			dnnForward (nodeArg);
			dnnBackward(nodeArg);
			dnnUpdate  (nodeArg);
		}

	}
	fprintf(cpuArg.pLogFile,"training over\n");
	fflush(cpuArg.pLogFile);

	WriteWts(nodeArg, cpuArg);
	gettimeofday(&timerStop, NULL);
	float timerElapsed = 1000.0 * (timerStop.tv_sec - timerStart.tv_sec) + (timerStop.tv_usec - timerStart.tv_usec) / 1000.0;
	fprintf(cpuArg.pLogFile,"total time cost: %fms\n", timerElapsed);
	fflush(cpuArg.pLogFile);

	UninitProgramConfig(cpuArg,nodeArg,oneChunk);

	return 0;
}
