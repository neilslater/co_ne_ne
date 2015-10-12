// ext/ru_ne_ne/struct_nn_model.c

#include "struct_nn_model.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for NNModel memory management
//

NNModel *nn_model__create() {
  NNModel *nn_model;
  nn_model = xmalloc( sizeof(NNModel) );
  nn_model->layers = NULL;
  nn_model->activations = NULL;
  nn_model->num_layers = 0;
  nn_model->num_inputs = 0;
  nn_model->num_outputs = 0;
  return nn_model;
}

void nn_model__destroy( NNModel *nn_model ) {
  int i;
  if ( nn_model->activations ) {
    for ( i = 0; i < nn_model->num_layers; i++ ) {
      xfree( nn_model->activations[i] );
    }
    xfree( nn_model->activations );
  }
  xfree( nn_model->layers );
  xfree( nn_model );
  return;
}

void nn_model__init( NNModel *nn_model, int num_layers, VALUE *layers ) {
  int i, last_num_outputs;
  Layer_FF *layer_ff;

  nn_model->num_layers = num_layers;
  nn_model->layers = ALLOC_N( VALUE, num_layers );
  nn_model->activations = ALLOC_N( float*, num_layers );
  // This immediate allocation avoids segfaults when cleaning up
  for ( i = 0; i < nn_model->num_layers; i++ ) {
    nn_model->activations[i] = NULL;
  }

  for ( i = 0; i < nn_model->num_layers; i++ ) {
    Data_Get_Struct( layers[i], Layer_FF, layer_ff );
    if ( i == 0 ) {
      nn_model->num_inputs = layer_ff->num_inputs;
    } else {
      if ( layer_ff->num_inputs != last_num_outputs ) {
        rb_raise( rb_eRuntimeError, "When building nn_model, layer connections failed between output size %d and next input size %d",
            last_num_outputs, layer_ff->num_inputs );
      }
    }
    last_num_outputs = layer_ff->num_outputs;

    nn_model->layers[i] = layers[i];
    nn_model->activations[i] = ALLOC_N( float, last_num_outputs );
  }

  nn_model->num_outputs = last_num_outputs;

  return;
}

void nn_model__gc_mark( NNModel *nn_model ) {
  int i;
  for ( i = 0; i < nn_model->num_layers; i++ ) {
    rb_gc_mark( nn_model->layers[i] );
  }
  return;
}

void nn_model__deep_copy( NNModel *nn_model_copy, NNModel *nn_model_orig ) {
  Layer_FF *layer_ff;

  nn_model_copy->num_layers = nn_model_orig->num_layers;
  nn_model_copy->num_inputs = nn_model_orig->num_inputs;
  nn_model_copy->num_outputs = nn_model_orig->num_outputs;

  nn_model_copy->layers = ALLOC_N( VALUE, nn_model_copy->num_layers );
  int i;
  for ( i = 0; i < nn_model_copy->num_layers; i++ ) {
    // This calls .clone of each layer via Ruby
    nn_model_copy->layers[i] = rb_funcall( nn_model_orig->layers[i], rb_intern("clone"), 0 );
  }

  nn_model_copy->activations = ALLOC_N( float*, nn_model_copy->num_layers );
  for ( i = 0; i < nn_model_copy->num_layers; i++ ) {
    nn_model_copy->activations[i] = NULL;
  }

  for ( i = 0; i < nn_model_copy->num_layers; i++ ) {
    layer_ff = nn_model__get_layer_ff_at( nn_model_copy, i );
    nn_model_copy->activations[i] = ALLOC_N( float, layer_ff->num_outputs );
    memcpy( nn_model_copy->activations[i], nn_model_orig->activations[i], layer_ff->num_outputs * sizeof(float) );
  }

  return;
}

NNModel * nn_model__clone( NNModel *nn_model_orig ) {
  NNModel * nn_model_copy = nn_model__create();
  nn_model__deep_copy( nn_model_copy, nn_model_orig );
  return nn_model_copy;
}

void nn_model__run( NNModel *nn_model, float *inputs ) {
  int i;

  layer_ff__run( nn_model__get_layer_ff_at( nn_model, 0 ),
      inputs, nn_model->activations[0] );

  for ( i = 1; i < nn_model->num_layers; i++ ) {
    // TODO: This only works for Layer_FF layers, we need a more flexible system
    layer_ff__run( nn_model__get_layer_ff_at( nn_model, i ),
        nn_model->activations[i-1], nn_model->activations[i] );
  }

  return;
}

Layer_FF *nn_model__get_layer_ff_at( NNModel *nn_model, int idx ) {
  Layer_FF * layer_ff;
  Data_Get_Struct( nn_model->layers[idx], Layer_FF, layer_ff );
  return layer_ff;
}

