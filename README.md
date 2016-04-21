# Deep Neural Network

This was an application that was run on Tianhe-2 at ASC 2016. We optimized using mpi and openmp. Take a look at [main.cpp](dnntk/main.cpp) for an example of mpi. Take a look at [dnn_kernel.cpp](dnntk/dnn_kernel.cpp) for an example of openmp.

### MPI

We farmed out the inner while loop in [main.cpp](dnntk/main.cpp) to a cluster of 8 nodes. The data is separated into chuncks with each chunck containing bunches. The chuncks we were working with each contained 100 bunches and 102400 samples meaning each bunch contained 1024 samples. On the second largest workload there was 525 bunches.

In the inner while loop after I got a chunck I solved for the number of bunches within that chunk with the line:

	int num_bunches = (readSize / cpuArg.bunchSize);

After I got the number of bunches, I divided up the work into equally sized samples for each node. `n` is the number of nodes in the cluster and `rank` is the id of the node. For example if you were running with 8 nodes `n = 8` and `rank` is in the range 0 to 7.

	/* divide loop */
	chunkstart = (num_bunches / n) * rank;
	if (num_bunches % n > rank){
		chunkstart += rank;
		chunkend = chunkstart + (num_bunches / n) + 1;
	}else{
		chunkstart += num_bunches % n;
		chunkend = chunkstart + (num_bunches / n);
  	}

I got the above code here: http://wiki.ccs.tulane.edu/index.php5/Parallel_Loop_MPI

Then I run the inner while loop and throw out the data I don't need in each node. `j` is a count of the bunch and it's in the range `0 <= j < num_bunches`.

	j = 0;
	// grab an nodeArg
	while (FetchOneBunch(oneChunk, nodeArg) == cpuArg.bunchSize) {
		
		if ((j >= chunkstart && j < chunkend) || ((rank == (n-1)) && j == chunkend)) {
			printf("train: %d on node %d \n", ++countCnt, rank);
			dnnForward (nodeArg);
			dnnBackward(nodeArg);
			dnnUpdate  (nodeArg);
		}
		j++;
	}

Each node runs and grabs all the data but only processes the data in it's range. The last node processes one more bunch than all the other nodes and is thus a bit behind. You could fix this by randomly assigning the last bunch instead of having the last node process it.

###

To make:

	cd dnntk_src
	make clean 
	make

To run locally:

	cd exp
	./train.sh

To run as a qsub job:	
	qsub: https://hpcc.usc.edu/support/documentation/running-a-job-on-the-hpcc-cluster-using-pbs/

