// ext/ru_ne_ne/struct_trainer_bp_layer.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definition for TrainerBPLayer and declarations for its memory management
//

#ifndef STRUCT_TRAINER_BP_LAYER_H
#define STRUCT_TRAINER_BP_LAYER_H

#include <ruby.h>
#include "narray.h"

typedef enum {GDACCEL_TYPE_NONE, GDACCEL_TYPE_MOMENTUM, GDACCEL_TYPE_RMSPROP} gd_accel_type;

typedef struct _trainer_bp_layer_raw {
  int num_inputs;
  int num_outputs;
  VALUE narr_de_dz;
  float *de_dz;
  VALUE narr_de_da;
  float *de_da;
  VALUE narr_de_dw;
  float *de_dw;
  VALUE narr_de_dw_momentum;
  float *de_dw_momentum;
  VALUE narr_de_dw_rmsprop;
  float *de_dw_rmsprop;
  float learning_rate;
  gd_accel_type gd_accel_type;
  float gd_accel_rate;
  float max_norm;
  float weight_decay;
  } TrainerBPLayer;

TrainerBPLayer *trainer_bp_layer__create();

void trainer_bp_layer__init( TrainerBPLayer *trainer_bp_layer, int num_inputs, int num_outputs );

void trainer_bp_layer__destroy( TrainerBPLayer *trainer_bp_layer );

void trainer_bp_layer__gc_mark( TrainerBPLayer *trainer_bp_layer );

void trainer_bp_layer__deep_copy( TrainerBPLayer *trainer_bp_layer_copy, TrainerBPLayer *trainer_bp_layer_orig );

TrainerBPLayer * trainer_bp_layer__clone( TrainerBPLayer *trainer_bp_layer_orig );

void trainer_bp_layer__start_batch( TrainerBPLayer *trainer_bp_layer );


#endif
