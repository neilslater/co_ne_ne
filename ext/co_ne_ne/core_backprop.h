// ext/con_ne_ne/core_backprop.h

#ifndef CORE_BACKPROP_H
#define CORE_BACKPROP_H

#include <ruby.h>
#include "narray.h"
#include "core_mt.h"
#include "core_narray.h"
#include <xmmintrin.h>

void activate_nn_layer_raw( int in_size, int out_size, float *in_ptr, float *weights, float *out_ptr );

float ms_error_raw( int out_size, float *out_ptr, float *target_ptr );

void calc_output_deltas_raw( int out_size, float *out_ptr, float *out_slope_ptr, float *target_ptr, float *out_delta_ptr );

void backprop_deltas_raw( int in_size, int out_size, float *in_deltas, float *in_slopes, float *weights, float *out_deltas );

void update_weights_raw( float eta, float m, int in_size, int out_size, float *inputs, float *weights, float *weights_last_deltas, float *output_deltas);

#endif
