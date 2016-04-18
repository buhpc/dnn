TK_INC_PATH		?= /opt/intel/mkl/include
TK_LIB_PATH		?= /opt/intel/mkl/lib/intel64
TK_LIB2_PATH		?= /opt/intel/lib/intel64

GCC					?= icpc

#flags
CCFLAGS				:=  -msse4.1 -O3 -g -p -Wall -fPIC

LDFLAGS				:=  -L$(TK_LIB_PATH) -L$(TK_LIB2_PATH) -ldl -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

INCLUDES			:= -I$(TK_INC_PATH) -I. 

all: clean build


build: dnntk

dnn_helper.o: dnn_helper.cpp
	$(GCC) $(CCFLAGS) $(INCLUDES)  -o $@ -c $<

dnn_func.o: dnn_func.cpp
	$(GCC) $(CCFLAGS) $(INCLUDES)  -o $@ -c $<

dnn_utility.o: dnn_utility.cpp
	$(GCC) $(CCFLAGS) $(INCLUDES)  -o $@ -c $<

main.o: main.cpp
	$(GCC) $(CCFLAGS) $(INCLUDES)  -o $@ -c $<

dnntk: dnn_helper.o dnn_kernel.o dnn_func.o dnn_utility.o  main.o
	$(GCC) $(CCFLAGS)  -o $@ $+ $(LDFLAGS)
	cp -f dnntk ../exp/       
clean:
	rm -f dnntk *.o *~ | clear
