// ext/ru_ne_ne/struct_nn_model.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definition for NNModel and declarations for its memory management
//

#ifndef STRUCT_NN_MODEL_H
#define STRUCT_NN_MODEL_H

#include <ruby.h>
#include "narray.h"
#include "struct_layer_ff.h"

typedef struct _nn_model_raw {
  VALUE *layers;
  float **activations;
  int num_layers;
  int num_inputs;
  int num_outputs;
  } NNModel;

NNModel *nn_model__create();

void nn_model__init( NNModel *nn_model, int num_layers, VALUE *layers );

void nn_model__destroy( NNModel *nn_model );

void nn_model__gc_mark( NNModel *nn_model );

void nn_model__deep_copy( NNModel *nn_model_copy, NNModel *nn_model_orig );

NNModel * nn_model__clone( NNModel *nn_model_orig );

#endif
