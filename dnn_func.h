#ifndef DNN_FUNC_H
#define DNN_FUNC_H

#include "dnn_utility.h"

extern "C" int dnnForward (NodeArg &nodeArg);
extern "C" int dnnBackward(NodeArg &nodeArg);
extern "C" int dnnUpdate  (NodeArg &nodeArg);

#endif
