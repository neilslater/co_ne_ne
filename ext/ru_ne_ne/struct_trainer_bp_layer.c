// ext/ru_ne_ne/struct_trainer_bp_layer.c

#include "struct_trainer_bp_layer.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for TrainerBPLayer memory management
//

TrainerBPLayer *trainer_bp_layer__create() {
  TrainerBPLayer *trainer_bp_layer;
  trainer_bp_layer = xmalloc( sizeof(TrainerBPLayer) );
  trainer_bp_layer->num_inputs = 0;
  trainer_bp_layer->num_outputs = 0;
  trainer_bp_layer->narr_de_dz = Qnil;
  trainer_bp_layer->de_dz = NULL;
  trainer_bp_layer->narr_de_da = Qnil;
  trainer_bp_layer->de_da = NULL;
  trainer_bp_layer->narr_de_dw = Qnil;
  trainer_bp_layer->de_dw = NULL;
  trainer_bp_layer->narr_de_dw_momentum = Qnil;
  trainer_bp_layer->de_dw_momentum = NULL;
  trainer_bp_layer->narr_de_dw_rmsprop = Qnil;
  trainer_bp_layer->de_dw_rmsprop = NULL;
  trainer_bp_layer->learning_rate = 0.01;
  trainer_bp_layer->smoothing_type = SMOOTH_TYPE_NONE;
  trainer_bp_layer->smoothing_rate = 0.9;
  trainer_bp_layer->max_norm = 0.0;
  trainer_bp_layer->weight_decay = 0.0;
  return trainer_bp_layer;
}

void trainer_bp_layer__init( TrainerBPLayer *trainer_bp_layer, int num_inputs, int num_outputs ) {
  int i;
  int shape[2];
  struct NARRAY *narr;
  float *narr_de_dz_ptr;
  float *narr_de_da_ptr;
  float *narr_de_dw_ptr;
  float *narr_de_dw_momentum_ptr;
  float *narr_de_dw_rmsprop_ptr;

  trainer_bp_layer->num_inputs = num_inputs;

  trainer_bp_layer->num_outputs = num_outputs;

  shape[0] = num_outputs;
  trainer_bp_layer->narr_de_dz = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( trainer_bp_layer->narr_de_dz, narr );
  narr_de_dz_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_dz_ptr[i] = 0.0;
  }
  trainer_bp_layer->de_dz = (float *) narr->ptr;

  shape[0] = num_inputs;
  trainer_bp_layer->narr_de_da = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( trainer_bp_layer->narr_de_da, narr );
  narr_de_da_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_da_ptr[i] = 0.0;
  }
  trainer_bp_layer->de_da = (float *) narr->ptr;

  shape[0] = num_inputs + 1;
  shape[1] = num_outputs;
  trainer_bp_layer->narr_de_dw = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( trainer_bp_layer->narr_de_dw, narr );
  narr_de_dw_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_dw_ptr[i] = 0.0;
  }
  trainer_bp_layer->de_dw = (float *) narr->ptr;

  trainer_bp_layer->narr_de_dw_momentum = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( trainer_bp_layer->narr_de_dw_momentum, narr );
  narr_de_dw_momentum_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_dw_momentum_ptr[i] = 0.0;
  }
  trainer_bp_layer->de_dw_momentum = (float *) narr->ptr;

  trainer_bp_layer->narr_de_dw_rmsprop = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( trainer_bp_layer->narr_de_dw_rmsprop, narr );
  narr_de_dw_rmsprop_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_dw_rmsprop_ptr[i] = 0.0;
  }
  trainer_bp_layer->de_dw_rmsprop = (float *) narr->ptr;

  return;
}

void trainer_bp_layer__destroy( TrainerBPLayer *trainer_bp_layer ) {
  xfree( trainer_bp_layer );
  return;
}

void trainer_bp_layer__gc_mark( TrainerBPLayer *trainer_bp_layer ) {
  rb_gc_mark( trainer_bp_layer->narr_de_dz );
  rb_gc_mark( trainer_bp_layer->narr_de_da );
  rb_gc_mark( trainer_bp_layer->narr_de_dw );
  rb_gc_mark( trainer_bp_layer->narr_de_dw_momentum );
  rb_gc_mark( trainer_bp_layer->narr_de_dw_rmsprop );
  return;
}

void trainer_bp_layer__deep_copy( TrainerBPLayer *trainer_bp_layer_copy, TrainerBPLayer *trainer_bp_layer_orig ) {
  struct NARRAY *narr;

  trainer_bp_layer_copy->num_inputs = trainer_bp_layer_orig->num_inputs;
  trainer_bp_layer_copy->num_outputs = trainer_bp_layer_orig->num_outputs;
  trainer_bp_layer_copy->learning_rate = trainer_bp_layer_orig->learning_rate;
  trainer_bp_layer_copy->smoothing_type = trainer_bp_layer_orig->smoothing_type;
  trainer_bp_layer_copy->smoothing_rate = trainer_bp_layer_orig->smoothing_rate;
  trainer_bp_layer_copy->max_norm = trainer_bp_layer_orig->max_norm;
  trainer_bp_layer_copy->weight_decay = trainer_bp_layer_orig->weight_decay;

  trainer_bp_layer_copy->narr_de_dz = na_clone( trainer_bp_layer_orig->narr_de_dz );
  GetNArray( trainer_bp_layer_copy->narr_de_dz, narr );
  trainer_bp_layer_copy->de_dz = (float *) narr->ptr;

  trainer_bp_layer_copy->narr_de_da = na_clone( trainer_bp_layer_orig->narr_de_da );
  GetNArray( trainer_bp_layer_copy->narr_de_da, narr );
  trainer_bp_layer_copy->de_da = (float *) narr->ptr;

  trainer_bp_layer_copy->narr_de_dw = na_clone( trainer_bp_layer_orig->narr_de_dw );
  GetNArray( trainer_bp_layer_copy->narr_de_dw, narr );
  trainer_bp_layer_copy->de_dw = (float *) narr->ptr;

  trainer_bp_layer_copy->narr_de_dw_momentum = na_clone( trainer_bp_layer_orig->narr_de_dw_momentum );
  GetNArray( trainer_bp_layer_copy->narr_de_dw_momentum, narr );
  trainer_bp_layer_copy->de_dw_momentum = (float *) narr->ptr;

  trainer_bp_layer_copy->narr_de_dw_rmsprop = na_clone( trainer_bp_layer_orig->narr_de_dw_rmsprop );
  GetNArray( trainer_bp_layer_copy->narr_de_dw_rmsprop, narr );
  trainer_bp_layer_copy->de_dw_rmsprop = (float *) narr->ptr;

  return;
}

TrainerBPLayer * trainer_bp_layer__clone( TrainerBPLayer *trainer_bp_layer_orig ) {
  TrainerBPLayer * trainer_bp_layer_copy = trainer_bp_layer__create();
  trainer_bp_layer__deep_copy( trainer_bp_layer_copy, trainer_bp_layer_orig );
  return trainer_bp_layer_copy;
}

void trainer_bp_layer__start_batch( TrainerBPLayer *trainer_bp_layer ) {
  int i,t = (trainer_bp_layer->num_inputs + 1 ) * trainer_bp_layer->num_outputs;
  float *de_dw = trainer_bp_layer->de_dw;
  for( i = 0; i < t; i++ ) {
    de_dw[i] = 0.0;
  }
  return;
}
