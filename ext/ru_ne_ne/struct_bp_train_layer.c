// ext/ru_ne_ne/struct_train_bp_layer.c

#include "struct_train_bp_layer.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating Train_BP_Layer structs
//

Train_BP_Layer *train_bp_layer__create() {
  Train_BP_Layer *train_bp_layer;
  train_bp_layer = xmalloc( sizeof(Train_BP_Layer) );

  train_bp_layer->rv_layer_ff = Qnil;
  train_bp_layer->narr_dE_dW = Qnil;
  train_bp_layer->narr_momentum_dE_dW = Qnil;
  train_bp_layer->narr_rmsprop_dE_dW = Qnil;

  // This is cached for performance when init_training is called
  train_bp_layer->layer_ff = NULL;

  // Training rate params
  train_bp_layer->lr = 0.001;
  train_bp_layer->momentum = 0.0;
  train_bp_layer->rmsprop = 1;
  train_bp_layer->rmsprop_adapt_rate = 0.1;

  // Regularisation params
  train_bp_layer->weight_decay = 0.0;
  train_bp_layer->max_norm = 0.0;

  return train_bp_layer;
}

void train_bp_layer__destroy( Train_BP_Layer *train_bp_layer ) {
  // Clear cache (not necessary)
  train_bp_layer->layer_ff = NULL;

  xfree( train_bp_layer );
  // No need to free NArrays - they will be handled by Ruby's GC, and may still be reachable
  return;
}

void train_bp_layer__gc_mark( Train_BP_Layer *train_bp_layer ) {
  rb_gc_mark( train_bp_layer->rv_layer_ff );
  rb_gc_mark( train_bp_layer->narr_dE_dW );
  rb_gc_mark( train_bp_layer->narr_momentum_dE_dW );
  rb_gc_mark( train_bp_layer->narr_rmsprop_dE_dW );
  return;
}

void train_bp_layer__init_training( Train_BP_Layer *train_bp_layer, VALUE rv_layer ) {
  int shape[2];
  struct NARRAY *narr;
  Layer_FF *layer_ff;

  // Check type of rv_layer, extract underlying layer_ff struct
  if ( TYPE(rv_layer) != T_DATA ||
      RDATA(rv_layer)->dfree != (RUBY_DATA_FUNC)layer_ff__destroy) {
    rb_raise( rb_eTypeError, "Expected a Layer object to initialise Training layer, but got something else" );
  }
  Data_Get_Struct( rv_layer, Layer_FF, layer_ff );

  train_bp_layer->rv_layer_ff = rv_layer;
  train_bp_layer->layer_ff = layer_ff;

  shape[0] = layer_ff->num_inputs + 1;
  shape[1] = layer_ff->num_outputs;

  train_bp_layer->narr_dE_dW = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( train_bp_layer->narr_dE_dW, narr );
  na_sfloat_set( narr->total, train_bp_layer->narr_dE_dW, (float) 0.0 );

  train_bp_layer->narr_momentum_dE_dW = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( train_bp_layer->narr_momentum_dE_dW, narr );
  na_sfloat_set( narr->total, train_bp_layer->narr_momentum_dE_dW, (float) 0.0 );

  train_bp_layer->narr_rmsprop_dE_dW = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( train_bp_layer->narr_rmsprop_dE_dW, narr );
  na_sfloat_set( narr->total, train_bp_layer->narr_rmsprop_dE_dW, (float) 0.0 );

  return;
}

void train_bp_layer__init_deltas( Train_BP_Layer *train_bp_layer ) {
  struct NARRAY *narr;
  GetNArray( train_bp_layer->narr_dE_dW, narr );
  na_sfloat_set( narr->total, train_bp_layer->narr_dE_dW, (float) 0.0 );
  return;
}

void train_bp_layer__propagate_deltas( Train_BP_Layer *train_bp_layer, float *input, float *output_delta ) {

  return;
}

void train_bp_layer__update_weights( Train_BP_Layer *train_bp_layer ) {

  return;
}
