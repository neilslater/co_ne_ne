// ext/ru_ne_ne/core_backprop.c

#include "core_backprop.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//   Definitions of C-only methods that calculate neural network activations and training results
//   from arrays of floats
//

void feed_forward_linear( int in_size, int out_size,
    float *in_ptr, float *weights, float *out_ptr ) {
  int i, j, in_aligned_size, offset;
  __m128 simd_x, simd_y, simd_t;
  float v[4];

  in_aligned_size = 4 * ( in_size/4 );

  // Calculate activation
  for ( i = 0; i < out_size; i++ ) {

    float t = 0.0;
    simd_t = _mm_setzero_ps();
    offset = i * (in_size + 1);

    // Use SIMD for all the aligned values in groups of 4
    for ( j = 0; j < in_aligned_size; j +=4 ) {
      // Unfortunately loadu is required, we just don't know the offset
      simd_x = _mm_loadu_ps( in_ptr + j );
      simd_y = _mm_loadu_ps( weights + (offset + j) );
      simd_x = _mm_mul_ps( simd_x, simd_y );
      simd_t = _mm_add_ps( simd_x, simd_t );
    }
    _mm_store_ps( v, simd_t );

    // Complete any remaining 1,2 or 3 items one at a time
    for ( j = in_aligned_size; j < in_size; j++ ) {
      t += in_ptr[ j ] * weights[ offset + j ];
    }

    // Add together 4 simd channels, plus bias
    out_ptr[i] = v[0] + v[1] + v[2] + v[3] + t + weights[ offset + in_size ];
  }

  return;
}

float mean_square_error( int out_size, float *out_ptr, float *target_ptr ) {
  int i;
  float t = 0.0;
  float d;

  for ( i = 0; i < out_size; i++ ) {
    d = out_ptr[i] - target_ptr[i];
    t += d * d;
  }

  return t/out_size;
}

// TODO: Needs to be more clearly split into dE_da vs dE_dz, and moved to trainer instead
//       of network
void core_calc_output_deltas( int out_size, float *out_ptr, float *out_slope_ptr,
      float *target_ptr, float *out_delta_ptr ) {
  int i;
  for ( i = 0; i < out_size; i++ ) {
    out_delta_ptr[i] = ( target_ptr[i] - out_ptr[i] ) * out_slope_ptr[i];
  }
  return;
}

// TODO: Needs to be more clearly split into dE_da vs dE_dz, and moved to trainer instead
//       of network
void core_backprop_deltas( int in_size, int out_size,
      float *in_deltas, float *in_slopes,
      float *weights, float *out_deltas ) {
  int i,j;
  float t;
  for( i = 0; i < in_size; i++ ) {
    t = 0.0;
    for( j = 0; j < out_size; j++ ) {
      t += weights[ j * (in_size+1) + i ] * out_deltas[j];
    }
    in_deltas[i] = t * in_slopes[i];
  }
  return;
}

// TODO: Needs to be moved to trainer instead of network, also we want to numerically
//       check all the gradients. This calculates dE_dW internally, but doesn't seem to
//       store it
void core_update_weights( float eta, float m, int in_size, int out_size,
        float *inputs, float *weights, float *weights_last_deltas, float *output_deltas) {

  int i,j, offset;
  float wupdate;

  // If j were the inner loop, this might be able to use SIMD
  for ( j = 0; j < out_size; j++ ) {
    offset = j * ( in_size + 1 );
    for ( i = 0; i < in_size; i++ ) {
      wupdate = (eta * output_deltas[j] * inputs[i]) + m * weights_last_deltas[ offset + i ];
      weights_last_deltas[ offset + i ] = wupdate;
      weights[ offset + i ] += wupdate;
    }
    // Update the bias value
    wupdate = (eta * output_deltas[j]) + m * weights_last_deltas[ offset + in_size ];
    weights_last_deltas[ offset + in_size ] = wupdate;
    weights[ offset + in_size ] += wupdate;
  }

  return;
}
