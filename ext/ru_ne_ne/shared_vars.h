// ext/ru_ne_ne/shared_vars.h

#ifndef SHARED_VARS_H
#define SHARED_VARS_H

#define NUM2FLT(x) ((float)NUM2DBL(x))
#define FLT2NUM(x) (rb_float_new((double)x))

// Force inclusion of hash declarations (only MRI includes by default)
#ifdef HAVE_RUBY_ST_H
#include "ruby/st.h"
#else
#include "st.h"
#endif

extern VALUE RuNeNe;
extern VALUE RuNeNe_Transfer;
extern VALUE RuNeNe_Transfer_Sigmoid;
extern VALUE RuNeNe_Transfer_TanH;
extern VALUE RuNeNe_Transfer_ReLU;
extern VALUE RuNeNe_Transfer_Linear;
extern VALUE RuNeNe_Transfer_Softmax;

extern VALUE RuNeNe_Objective;
extern VALUE RuNeNe_Objective_MeanSquaredError;
extern VALUE RuNeNe_Objective_LogLoss;
extern VALUE RuNeNe_Objective_MulticlassLogLoss;

extern VALUE RuNeNe_GradientDescent;
extern VALUE RuNeNe_GradientDescent_SGD;
extern VALUE RuNeNe_GradientDescent_NAG;
extern VALUE RuNeNe_GradientDescent_RMSProp;

extern VALUE RuNeNe_Layer;
extern VALUE RuNeNe_Layer_FeedForward;
extern VALUE RuNeNe_Network;
extern VALUE RuNeNe_DataSet;

extern VALUE RuNeNe_Learn;
extern VALUE RuNeNe_Learn_MBGD;
extern VALUE RuNeNe_Learn_MBGD_Layer;
extern VALUE RuNeNe_Trainer_BPNetwork;

#endif
