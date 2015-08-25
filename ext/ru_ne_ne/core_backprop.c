// ext/ru_ne_ne/core_backprop.c

#include "core_backprop.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//   Definitions of C-only methods that calculate neural network activations and training results
//   from arrays of floats
//


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
