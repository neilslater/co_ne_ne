// ext/ru_ne_ne/shared_vars.h

#ifndef SHARED_VARS_H
#define SHARED_VARS_H

#define NUM2FLT(x) ((float)NUM2DBL(x))
#define FLT2NUM(x) (rb_float_new((double)x))

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

extern VALUE RuNeNe_Layer;
extern VALUE RuNeNe_Layer_FeedForward;
extern VALUE RuNeNe_Network;
extern VALUE RuNeNe_TrainingData;
extern VALUE RuNeNe_Trainer;
extern VALUE RuNeNe_Trainer_BPLayer;
extern VALUE RuNeNe_Trainer_BPNetwork;

#endif
