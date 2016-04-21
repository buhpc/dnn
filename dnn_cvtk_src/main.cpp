#include "dnn_func.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
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

	// Get the number of processes
	int n;
	MPI_Comm_size(MPI_COMM_WORLD, &n);
	printf("Number of nodes %d", n);

	// Get the rank of the process
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// j is the chunck count 
	int j;

	while ((readSize = FetchOneChunk(cpuArg, oneChunk)) && readSize) {
		fprintf(cpuArg.pLogFile,"--chunk(%d) : containing samples %d\n", chunkCnt++, readSize);

		fflush(cpuArg.pLogFile);

		// printf("readSize = %d\n", readSize);
		
		// printf("Number of nodes (n) = %d ... rank = %d \n", n, rank);

		// Start the chunck and end it
		int chunkstart;
		int chunkend;

		int num_bunches = (readSize / cpuArg.bunchSize);

		// printf("num_bunches = %d\n", num_bunches);

		/* divide loop */
		chunkstart = (num_bunches / n) * rank;
		if (num_bunches % n > rank){
			chunkstart += rank;
			chunkend = chunkstart + (num_bunches / n) + 1;
		}else{
			chunkstart += num_bunches % n;
			chunkend = chunkstart + (num_bunches / n);
	  	}

	  	// printf("chunkstart = %d <---> chunkend = %d \n", chunkstart, chunkend);

	  	// int read = FetchOneBunch(oneChunk, nodeArg);

	  	// printf("cpuArg.bunchSize = %d\n", cpuArg.bunchSize);
	  	j = 0;
		while (FetchOneBunch(oneChunk, nodeArg) == cpuArg.bunchSize) {
			
			// grab an nodeArg

			// printf("j = %d\n", j);
			
			if ((j >= chunkstart && j < chunkend) || ((rank == (n-1)) && j == chunkend)) {
				printf("train: %d on node %d \n", ++countCnt, rank);
				dnnForward (nodeArg);
				dnnBackward(nodeArg);
				dnnUpdate  (nodeArg);
			}

			//read = FetchOneBunch(oneChunk, nodeArg);

			j++;

			// end:
	  		//	goto end;
		}

		//printf ("j = %d\n", j);

		//			 Original code	
//		while (FetchOneBunch(oneChunk, nodeArg) == cpuArg.bunchSize) {
//			MPI_Barrier(MPI_COMM_WORLD);
//			printf("train: %d \n", ++countCnt);
//			dnnForward (nodeArg);
//			dnnBackward(nodeArg);
//			dnnUpdate  (nodeArg);
//		}

	}
	// Finalize the MPI environment.
	MPI_Finalize();

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
