#!/bin/bash
###specify the queue of job
#PBS -q mic

###specify the name of job
#PBS -N mic_para


###Submit to specified nodes: If mic program , use 24 cores on one node.
#PBS -l nodes=1:ppn=24

###prepare env for computing

export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mic/coi/host-linux-release/lib/

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > $PBS_O_WORKDIR/hosts.${PBS_JOBID}
NP=`cat $PBS_NODEFILE | wc -l`
###just replace mic_program with your input file,optional argument lists. for mpi
mpirun -np $NP -machinefile $PBS_O_WORKDIR/hosts.${PBS_JOBID} ./intro_sampleC.out

###If openmp
#./mic_program

rm -f $PBS_O_WORKDIR/hosts.${PBS_JOBID}