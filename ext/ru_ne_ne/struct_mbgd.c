// ext/ru_ne_ne/struct_mbgd.c

#include "struct_mbgd.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for MBGD memory management
//

MBGD *mbgd__create() {
  MBGD *mbgd;
  mbgd = xmalloc( sizeof(MBGD) );
  mbgd->mbgd_layers = NULL;
  mbgd->num_layers = 0;
  mbgd->num_inputs = 0;
  mbgd->num_outputs = 0;
  mbgd->objective = MSE;
  return mbgd;
}

void mbgd__init( MBGD *mbgd, int num_mbgd_layers, VALUE *mbgd_layers ) {
  int i, last_num_outputs;
  MBGDLayer *mbgd_layer;

  mbgd->num_layers = num_mbgd_layers;
  mbgd->mbgd_layers = ALLOC_N( VALUE, num_mbgd_layers );

  for ( i = 0; i < mbgd->num_layers; i++ ) {
    Data_Get_Struct( mbgd_layers[i], MBGDLayer, mbgd_layer );
    if ( i == 0 ) {
      mbgd->num_inputs = mbgd_layer->num_inputs;
    } else {
      if ( mbgd_layer->num_inputs != last_num_outputs ) {
        rb_raise( rb_eRuntimeError, "When building mbgd, layer connections failed between output size %d and next input size %d",
            last_num_outputs, mbgd_layer->num_inputs );
      }
    }
    last_num_outputs = mbgd_layer->num_outputs;

    mbgd->mbgd_layers[i] = mbgd_layers[i];
  }

  mbgd->num_outputs = last_num_outputs;

  return;
}


void mbgd__destroy( MBGD *mbgd ) {
  xfree( mbgd->mbgd_layers );
  xfree( mbgd );
  return;
}

void mbgd__gc_mark( MBGD *mbgd ) {
  int i;
  for ( i = 0; i < mbgd->num_layers; i++ ) {
    rb_gc_mark( mbgd->mbgd_layers[i] );
  }
  return;
}

void mbgd__deep_copy( MBGD *mbgd_copy, MBGD *mbgd_orig ) {
  mbgd_copy->num_layers = mbgd_orig->num_layers;
  mbgd_copy->num_inputs = mbgd_orig->num_inputs;
  mbgd_copy->num_outputs = mbgd_orig->num_outputs;

  mbgd_copy->mbgd_layers = ALLOC_N( VALUE, mbgd_copy->num_layers );
  int i;
  for ( i = 0; i < mbgd_copy->num_layers; i++ ) {
    // This calls .clone of each layer via Ruby
    mbgd_copy->mbgd_layers[i] = rb_funcall( mbgd_orig->mbgd_layers[i], rb_intern("clone"), 0 );
  }

  return;
}

MBGD * mbgd__clone( MBGD *mbgd_orig ) {
  MBGD * mbgd_copy = mbgd__create();
  mbgd__deep_copy( mbgd_copy, mbgd_orig );
  return mbgd_copy;
}

MBGDLayer *mbgd__get_mbgd_layer_at( MBGD *mbgd, int idx ) {
  MBGDLayer * mbgd_layer;
  Data_Get_Struct( mbgd->mbgd_layers[idx], MBGDLayer, mbgd_layer );
  return mbgd_layer;
}

void mbgd__check_size_compatible( MBGD *mbgd, NNModel *nn_model, DataSet *dataset ) {
  int i;
  Layer_FF * layer_ff;
  MBGDLayer * mbgd_layer;

  if ( mbgd->num_inputs != nn_model->num_inputs || dataset->input_item_size != nn_model->num_inputs ) {
    rb_raise( rb_eArgError, "Input size mismatch. Dataset %d, NNModel %d, MBGD %d.",  dataset->input_item_size, nn_model->num_inputs, mbgd->num_inputs );
  }

  if ( mbgd->num_outputs != nn_model->num_outputs || dataset->output_item_size != nn_model->num_outputs ) {
    rb_raise( rb_eArgError, "Output size mismatch. Dataset %d, NNModel %d, MBGD %d.",  dataset->output_item_size, nn_model->num_outputs, mbgd->num_outputs );
  }

  if ( mbgd->num_layers != nn_model->num_layers ) {
    rb_raise( rb_eArgError, "Number of layers mismatch. NNModel %d, MBGD %d.",  nn_model->num_layers, mbgd->num_layers );
  }

  for( i = 0; i < mbgd->num_layers; i++ ) {
    layer_ff = nn_model__get_layer_ff_at( nn_model, i );
    mbgd_layer = mbgd__get_mbgd_layer_at( mbgd, i );

    if ( layer_ff->num_inputs != mbgd_layer->num_inputs || mbgd_layer->num_outputs != layer_ff->num_outputs ) {
      rb_raise( rb_eArgError, "Layer size mismatch in layer %d. NNModel %d in, % d out. MBGD %d in, %d out.",
        i, layer_ff->num_inputs, layer_ff->num_outputs, mbgd_layer->num_inputs, mbgd_layer->num_outputs );
    }
  }

  return;
}


float mbgd__train_one_batch( MBGD *mbgd, NNModel *nn_model, DataSet *dataset, objective_type o, int batch_size ) {
  int i, j;
  float o_score = 0.0;
  float *inputs;
  float *targets;
  float *predictions;
  float *layer_activations;
  float *layer_inputs;

  Layer_FF * layer_ff;
  MBGDLayer * mbgd_layer;
  MBGDLayer * upper_mbgd_layer;

  // Start batch each layer pair
  for ( i = 0; i < mbgd->num_layers; i++ ) {
    mbgd_layer__start_batch(
      mbgd__get_mbgd_layer_at( mbgd, i ),
      nn_model__get_layer_ff_at( nn_model, i ) );
  }

  for ( i = 0; i < batch_size; i++ ) {
    // Get next dataset entry
    inputs = dataset__current_input( dataset );
    targets = dataset__current_output( dataset );

    // Run through network
    nn_model__run( nn_model, inputs );
    predictions = nn_model->activations[ nn_model->num_layers - 1 ];

    // Collect objective function score (TODO: This should use instance objective)
    o_score += objective_function_loss( o, mbgd->num_outputs, predictions, targets );

    // Back-propagate error to output layer (TODO: This should use instance objective)
    mbgd_layer = mbgd__get_mbgd_layer_at( mbgd, mbgd->num_layers - 1 );
    layer_ff = nn_model__get_layer_ff_at( nn_model, mbgd->num_layers - 1 );
    layer_activations = nn_model->activations[ mbgd->num_layers - 1 ];
    if ( mbgd->num_layers > 1 ) {
      layer_inputs = nn_model->activations[ mbgd->num_layers - 2 ];
    } else {
      layer_inputs = inputs;
    }

    mbgd_layer__backprop_for_output_layer( mbgd_layer, layer_ff,
        layer_inputs, layer_activations, targets, o );

    // Continue back-propagation to all earlier layers
    for ( j = mbgd->num_layers - 2; j >= 0; j-- ) {
      upper_mbgd_layer = mbgd_layer;
      mbgd_layer = mbgd__get_mbgd_layer_at( mbgd, j );
      layer_ff = nn_model__get_layer_ff_at( nn_model, j );
      layer_activations =  nn_model->activations[ j ];
      if ( j > 0 ) {
        layer_inputs = nn_model->activations[ j - 1 ];
      } else {
        layer_inputs = inputs;
      }

      // FIXME: this needlessly calculates de_da for input layer
      mbgd_layer__backprop_for_mid_layer( mbgd_layer, layer_ff,
        layer_inputs, layer_activations, upper_mbgd_layer->de_da );
    }

    // Next item
    dataset__next( dataset );
  }

  // Weight update each layer pair
  for ( i = 0; i < mbgd->num_layers; i++ ) {
    mbgd_layer__finish_batch(
      mbgd__get_mbgd_layer_at( mbgd, i ),
      nn_model__get_layer_ff_at( nn_model, i ) );
  }

  return o_score / batch_size;
}
