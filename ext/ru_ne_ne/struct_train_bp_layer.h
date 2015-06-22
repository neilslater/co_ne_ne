// ext/ru_ne_ne/struct_train_bp_layer.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating Train_BP_Layer structs
//

#ifndef STRUCT_TRAIN_BP_LAYER_H
#define STRUCT_TRAIN_BP_LAYER_H

#include <ruby.h>
#include "narray.h"
#include "struct_layer_ff.h"
#include "core_backprop.h"

typedef struct _train_bp_layer_raw {
    VALUE rv_layer_ff;          // RuNeNe_Layer_FeedForward object that is being trained
    VALUE narr_dE_dW;           // Sum of delta back-propagated to weights (per batch)
    VALUE narr_momentum_dE_dW;  // Per-batch momentum term
    VALUE narr_rmsprop_dE_dW;   // Per-batch rmsprop weight update factor

    // This is cached for performance when init_training is called
    Layer_FF *layer_ff;

    // Training rate params
    float lr;
    float momentum;
    bool rmsprop;
    float rmsprop_adapt_rate;

    // Regularisation params
    float weight_decay;
    float max_norm;
  } Train_BP_Layer;

Train_BP_Layer *train_bp_layer__create();

void train_bp_layer__destroy( Train_BP_Layer *train_bp_layer );

void train_bp_layer__gc_mark( Train_BP_Layer *train_bp_layer );

void train_bp_layer__init_training( Train_BP_Layer *train_bp_layer, VALUE rv_layer );

void train_bp_layer__init_deltas( Train_BP_Layer *train_bp_layer );

void train_bp_layer__propagate_deltas( Train_BP_Layer *train_bp_layer, float *input, float *output_delta );

void train_bp_layer__update_weights( Train_BP_Layer *train_bp_layer );

#endif
