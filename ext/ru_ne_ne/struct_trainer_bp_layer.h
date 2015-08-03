// ext/ru_ne_ne/struct_trainer_bp_layer.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definition for TrainerBPLayer and declarations for its memory management
//

#ifndef STRUCT_TRAINER_BP_LAYER_H
#define STRUCT_TRAINER_BP_LAYER_H

#include <ruby.h>
#include "narray.h"

typedef enum {SMOOTH_TYPE_NONE, SMOOTH_TYPE_MOMENTUM, SMOOTH_TYPE_RMSPROP} bp_smooth_type;

typedef struct _trainer_bp_layer_raw {
  int num_inputs;
  int num_outputs;
  int *de_dz_shape;
  VALUE narr_de_dz;
  float *de_dz;
  int *de_da_shape;
  VALUE narr_de_da;
  float *de_da;
  int *de_dw_shape;
  VALUE narr_de_dw;
  float *de_dw;
  VALUE narr_de_dw_momentum;
  float *de_dw_momentum;
  VALUE narr_de_dw_rmsprop;
  float *de_dw_rmsprop;
  float learning_rate;
  bp_smooth_type smoothing_type;
  float smoothing_rate;
  float max_norm;
  float weight_decay;
  } TrainerBPLayer;

TrainerBPLayer *trainer_bp_layer__create();

void trainer_bp_layer__init( TrainerBPLayer *trainer_bp_layer, int num_inputs, int num_outputs );

void trainer_bp_layer__destroy( TrainerBPLayer *trainer_bp_layer );

void trainer_bp_layer__gc_mark( TrainerBPLayer *trainer_bp_layer );

void trainer_bp_layer__deep_copy( TrainerBPLayer *trainer_bp_layer_copy, TrainerBPLayer *trainer_bp_layer_orig );

TrainerBPLayer * trainer_bp_layer__clone( TrainerBPLayer *trainer_bp_layer_orig );

#endif
