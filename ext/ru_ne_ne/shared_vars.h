// ext/ru_ne_ne/shared_vars.h

#ifndef SHARED_VARS_H
#define SHARED_VARS_H

#include "shared_helpers.h"

extern volatile VALUE RuNeNe;
extern volatile VALUE RuNeNe_Transfer;
extern volatile VALUE RuNeNe_Transfer_Sigmoid;
extern volatile VALUE RuNeNe_Transfer_TanH;
extern volatile VALUE RuNeNe_Transfer_ReLU;
extern volatile VALUE RuNeNe_Transfer_Linear;
extern volatile VALUE RuNeNe_Transfer_Softmax;

extern volatile VALUE RuNeNe_Objective;
extern volatile VALUE RuNeNe_Objective_MeanSquaredError;
extern volatile VALUE RuNeNe_Objective_LogLoss;
extern volatile VALUE RuNeNe_Objective_MulticlassLogLoss;

extern volatile VALUE RuNeNe_GradientDescent;
extern volatile VALUE RuNeNe_GradientDescent_SGD;
extern volatile VALUE RuNeNe_GradientDescent_NAG;
extern volatile VALUE RuNeNe_GradientDescent_RMSProp;

extern volatile VALUE RuNeNe_Layer;
extern volatile VALUE RuNeNe_Layer_FeedForward;

extern volatile VALUE RuNeNe_NNModel;

extern volatile VALUE RuNeNe_DataSet;

extern volatile VALUE RuNeNe_Learn;
extern volatile VALUE RuNeNe_Learn_MBGD;
extern volatile VALUE RuNeNe_Learn_MBGD_Layer;

#endif
