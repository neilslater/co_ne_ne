// ext/con_ne_ne/core_backprop.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//   Declarations of C-only methods that calculate neural network activations and training results
//   from arrays of floats
//

#ifndef CORE_BACKPROP_H
#define CORE_BACKPROP_H

#include "core_mt.h"
#include "core_narray.h"
#include <xmmintrin.h>

void core_activate_layer_output( int in_size, int out_size, float *in_ptr, float *weights, float *out_ptr );

float core_mean_square_error( int out_size, float *out_ptr, float *target_ptr );

void core_calc_output_deltas( int out_size, float *out_ptr, float *out_slope_ptr, float *target_ptr, float *out_delta_ptr );

void core_backprop_deltas( int in_size, int out_size, float *in_deltas, float *in_slopes, float *weights, float *out_deltas );

void core_update_weights( float eta, float m, int in_size, int out_size, float *inputs, float *weights, float *weights_last_deltas, float *output_deltas);

#endif
