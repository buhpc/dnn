#!/bin/bash

###specify the queue of job
#PBS -q cpu

###specify the name of job
#PBS -N cpu_para

###Submit to specified 2 nodes and 4 cores per node
#PBS -l nodes=2:ppn=4

###prepare env for computing

export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mic/coi/host-linux-release/lib/

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > $PBS_O_WORKDIR/hosts.${PBS_JOBID}
NP=`cat $PBS_NODEFILE | wc -l`
###just replace cpi-mpich with your input file,optional argument lists.

mpirun -np 24 -machinefile $PBS_O_WORKDIR/hosts.${PBS_JOBID} ./dnntk dnn_config_debug.cfg


rm -f $PBS_O_WORKDIR/hosts.${PBS_JOBID}
