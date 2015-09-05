// ext/ru_ne_ne/struct_mbgd_layer.c

#include "struct_mbgd_layer.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for MBGDLayer memory management
//

MBGDLayer *mbgd_layer__create() {
  MBGDLayer *mbgd_layer;
  mbgd_layer = xmalloc( sizeof(MBGDLayer) );
  mbgd_layer->num_inputs = 0;
  mbgd_layer->num_outputs = 0;
  mbgd_layer->narr_de_dz = Qnil;
  mbgd_layer->de_dz = NULL;
  mbgd_layer->narr_de_da = Qnil;
  mbgd_layer->de_da = NULL;
  mbgd_layer->narr_de_dw = Qnil;
  mbgd_layer->de_dw = NULL;
  mbgd_layer->narr_de_dw_stats_a = Qnil;
  mbgd_layer->de_dw_stats_a = NULL;
  mbgd_layer->narr_de_dw_stats_b = Qnil;
  mbgd_layer->de_dw_stats_b = NULL;
  mbgd_layer->learning_rate = 0.01;
  mbgd_layer->gd_accel_type = GDACCEL_TYPE_NONE;
  mbgd_layer->gd_accel_rate = 0.9;
  mbgd_layer->max_norm = 0.0;
  mbgd_layer->weight_decay = 0.0;
  return mbgd_layer;
}

void mbgd_layer__init( MBGDLayer *mbgd_layer, int num_inputs, int num_outputs ) {
  int i;
  int shape[2];
  struct NARRAY *narr;
  float *narr_de_dz_ptr;
  float *narr_de_da_ptr;
  float *narr_de_dw_ptr;
  float *narr_de_dw_stats_a_ptr;
  float *narr_de_dw_stats_b_ptr;

  mbgd_layer->num_inputs = num_inputs;

  mbgd_layer->num_outputs = num_outputs;

  shape[0] = num_outputs;
  mbgd_layer->narr_de_dz = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( mbgd_layer->narr_de_dz, narr );
  narr_de_dz_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_dz_ptr[i] = 0.0;
  }
  mbgd_layer->de_dz = (float *) narr->ptr;

  shape[0] = num_inputs;
  mbgd_layer->narr_de_da = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( mbgd_layer->narr_de_da, narr );
  narr_de_da_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_da_ptr[i] = 0.0;
  }
  mbgd_layer->de_da = (float *) narr->ptr;

  shape[0] = num_inputs + 1;
  shape[1] = num_outputs;
  mbgd_layer->narr_de_dw = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( mbgd_layer->narr_de_dw, narr );
  narr_de_dw_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_dw_ptr[i] = 0.0;
  }
  mbgd_layer->de_dw = (float *) narr->ptr;

  mbgd_layer->narr_de_dw_stats_a = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( mbgd_layer->narr_de_dw_stats_a, narr );
  narr_de_dw_stats_a_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_dw_stats_a_ptr[i] = 0.0;
  }
  mbgd_layer->de_dw_stats_a = (float *) narr->ptr;

  mbgd_layer->narr_de_dw_stats_b = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( mbgd_layer->narr_de_dw_stats_b, narr );
  narr_de_dw_stats_b_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_de_dw_stats_b_ptr[i] = 0.0;
  }
  mbgd_layer->de_dw_stats_b = (float *) narr->ptr;

  return;
}

void mbgd_layer__destroy( MBGDLayer *mbgd_layer ) {
  xfree( mbgd_layer );
  return;
}

void mbgd_layer__gc_mark( MBGDLayer *mbgd_layer ) {
  rb_gc_mark( mbgd_layer->narr_de_dz );
  rb_gc_mark( mbgd_layer->narr_de_da );
  rb_gc_mark( mbgd_layer->narr_de_dw );
  rb_gc_mark( mbgd_layer->narr_de_dw_stats_a );
  rb_gc_mark( mbgd_layer->narr_de_dw_stats_b );
  return;
}

void mbgd_layer__deep_copy( MBGDLayer *mbgd_layer_copy, MBGDLayer *mbgd_layer_orig ) {
  struct NARRAY *narr;

  mbgd_layer_copy->num_inputs = mbgd_layer_orig->num_inputs;
  mbgd_layer_copy->num_outputs = mbgd_layer_orig->num_outputs;
  mbgd_layer_copy->learning_rate = mbgd_layer_orig->learning_rate;
  mbgd_layer_copy->gd_accel_type = mbgd_layer_orig->gd_accel_type;
  mbgd_layer_copy->gd_accel_rate = mbgd_layer_orig->gd_accel_rate;
  mbgd_layer_copy->max_norm = mbgd_layer_orig->max_norm;
  mbgd_layer_copy->weight_decay = mbgd_layer_orig->weight_decay;

  mbgd_layer_copy->narr_de_dz = na_clone( mbgd_layer_orig->narr_de_dz );
  GetNArray( mbgd_layer_copy->narr_de_dz, narr );
  mbgd_layer_copy->de_dz = (float *) narr->ptr;

  mbgd_layer_copy->narr_de_da = na_clone( mbgd_layer_orig->narr_de_da );
  GetNArray( mbgd_layer_copy->narr_de_da, narr );
  mbgd_layer_copy->de_da = (float *) narr->ptr;

  mbgd_layer_copy->narr_de_dw = na_clone( mbgd_layer_orig->narr_de_dw );
  GetNArray( mbgd_layer_copy->narr_de_dw, narr );
  mbgd_layer_copy->de_dw = (float *) narr->ptr;

  mbgd_layer_copy->narr_de_dw_stats_a = na_clone( mbgd_layer_orig->narr_de_dw_stats_a );
  GetNArray( mbgd_layer_copy->narr_de_dw_stats_a, narr );
  mbgd_layer_copy->de_dw_stats_a = (float *) narr->ptr;

  mbgd_layer_copy->narr_de_dw_stats_b = na_clone( mbgd_layer_orig->narr_de_dw_stats_b );
  GetNArray( mbgd_layer_copy->narr_de_dw_stats_b, narr );
  mbgd_layer_copy->de_dw_stats_b = (float *) narr->ptr;

  return;
}

MBGDLayer * mbgd_layer__clone( MBGDLayer *mbgd_layer_orig ) {
  MBGDLayer * mbgd_layer_copy = mbgd_layer__create();
  mbgd_layer__deep_copy( mbgd_layer_copy, mbgd_layer_orig );
  return mbgd_layer_copy;
}

void mbgd_layer__start_batch( MBGDLayer *mbgd_layer, Layer_FF *layer_ff ) {
  int i,t = (mbgd_layer->num_inputs + 1 ) * mbgd_layer->num_outputs;

  // Re-set accumulated de_dw for this batch
  float *de_dw = mbgd_layer->de_dw;
  float *weights = layer_ff->weights;
  float *weight_vel = mbgd_layer->de_dw_stats_a;
  float momentum = mbgd_layer->gd_accel_rate;

  for( i = 0; i < t; i++ ) {
    de_dw[i] = 0.0;
  }

  switch ( mbgd_layer->gd_accel_type ) {
    case GDACCEL_TYPE_NONE:

      break;

    case GDACCEL_TYPE_MOMENTUM:
      // Move layer weights forward so we assess gradient at weights plus momentum * mf
      for( i = 0; i < t; i++ ) {
        weight_vel[i] *= momentum;
        weights[i] += weight_vel[i];
      }
      break;

    case GDACCEL_TYPE_RMSPROP:

      break;
  }
  return;
}

void increment_de_dw_from_de_dz( int in_size, int out_size, float *inputs, float *de_dw, float *de_dz) {
  int i,j, offset;

  // If j were the inner loop, this might be able to use SIMD
  for ( j = 0; j < out_size; j++ ) {
    offset = j * ( in_size + 1 );
    for ( i = 0; i < in_size; i++ ) {
      de_dw[ offset + i ] += de_dz[j] * inputs[i];
      // TODO: We could do de_da[i] += de_dz[j] * weights[ offset + i ]; here?
      // measure best combination . . .
    }
    // For the bias, we have no input value
    de_dw[ offset + in_size ] += de_dz[j];
  }

  return;
}

// This "drops down" one layer calculating de_da for *inputs* to a layer
void calc_de_da_from_de_dz( int in_size, int out_size, float *weights, float *de_da, float *de_dz) {
  int i,j;
  float t;

  for ( i = 0; i < in_size; i++ ) {
    t = 0.0;
    for ( j = 0; j < out_size; j++ ) {
      t += de_dz[j] * weights[ j * ( in_size + 1 ) + i ];
    }
    de_da[i] = t;
  }

  return;
}

void mbgd_layer__backprop_for_output_layer( MBGDLayer *mbgd_layer, Layer_FF *layer_ff,
      float *input, float *output, float *target, objective_type o ) {

  de_dz_from_objective_and_transfer( o,
      layer_ff->transfer_fn,
      mbgd_layer->num_outputs,
      output,
      target,
      mbgd_layer->de_dz );

  increment_de_dw_from_de_dz( mbgd_layer->num_inputs,
      mbgd_layer->num_outputs,
      input,
      mbgd_layer->de_dw,
      mbgd_layer->de_dz );

  // TODO: Either combine for speed with incr_de_dw *or* make it optional (not required in first layer)
  calc_de_da_from_de_dz( mbgd_layer->num_inputs,
      mbgd_layer->num_outputs,
      layer_ff->weights,
      mbgd_layer->de_da,
      mbgd_layer->de_dz );

  return;
}

void  de_dz_from_upper_de_da( transfer_type t, int out_size, float *output, float *de_da, float *de_dz ) {
  int i,k;
  float tmp;

  // TODO: Optimise this away if not required (a mid-layer softmax is unusual)
  if ( t == SOFTMAX ) {
    // Generic softmax da_dz needs larger storage array
    float *da_dz = xmalloc( sizeof(float) * out_size * out_size );

    raw_softmax_bulk_derivative_at( out_size, output, da_dz );

    for ( i = 0; i < out_size ; i++ ) {
      tmp = 0.0;
      for ( k = 0; k < out_size ; k++ ) {
        tmp += de_da[k] * da_dz[i * out_size + k];
      }
      de_dz[i] = tmp;
    }
    xfree( da_dz );
  } else {
    // This stores da_dz . . .
    transfer_bulk_derivative_at( t, out_size, output, de_dz );
    // de_dz = de_da * da_dz
    for ( i = 0; i < out_size; i++ ) {
      de_dz[i] *= de_da[i];
    }
  }

  return;
}

void mbgd_layer__backprop_for_mid_layer( MBGDLayer *mbgd_layer, Layer_FF *layer_ff,
      float *input, float *output, float *upper_de_da ) {

  de_dz_from_upper_de_da( layer_ff->transfer_fn,
      mbgd_layer->num_outputs,
      output,
      upper_de_da,
      mbgd_layer->de_dz );

  increment_de_dw_from_de_dz( mbgd_layer->num_inputs,
      mbgd_layer->num_outputs,
      input,
      mbgd_layer->de_dw,
      mbgd_layer->de_dz );

  // TODO: Either combine for speed with incr_de_dw *or* make it optional (not required in first layer)
  calc_de_da_from_de_dz( mbgd_layer->num_inputs,
      mbgd_layer->num_outputs,
      layer_ff->weights,
      mbgd_layer->de_da,
      mbgd_layer->de_dz );

  return;
}

void mbgd_layer__finish_batch( MBGDLayer *mbgd_layer, Layer_FF *layer_ff ) {
  // Needs momentum and rmsprop adding . . .

  int i, t;
  t = (mbgd_layer->num_inputs + 1) * mbgd_layer->num_outputs;
  float lr = mbgd_layer->learning_rate;
  float *weights = layer_ff->weights;
  float *de_dw = mbgd_layer->de_dw;

  float *weight_vel;

  switch ( mbgd_layer->gd_accel_type ) {
    case GDACCEL_TYPE_NONE:
      for( i = 0; i < t; i++ ) {
          weights[i] -= de_dw[i] * lr;
      }
      break;

    case GDACCEL_TYPE_MOMENTUM:
      weight_vel = mbgd_layer->de_dw_stats_a;

      for( i = 0; i < t; i++ ) {
        weight_vel[i] -= lr * de_dw[i];
        weights[i] -= lr * de_dw[i];
      }
      break;

    case GDACCEL_TYPE_RMSPROP:
      // TODO: Implement RMSPROP
      for( i = 0; i < t; i++ ) {
        weights[i] -= de_dw[i] * lr;
      }

      break;
  }

  return;
}
