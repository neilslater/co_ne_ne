// ext/ru_ne_ne/core_regularise.c

#include "core_regularise.h"

void apply_weight_decay( int num_inputs, int num_outputs, float *weights, float *de_dw, float weight_decay ) {
  int i, j, offset;
  for ( j = 0; j < num_outputs; j++ ) {
    offset = ( num_inputs + 1 ) * j;
    for ( i = 0; i < num_inputs; i++ ) {
      de_dw[ offset + i ] += weights[ offset + i ] * weight_decay;
    }
  }
  return;
}

void apply_max_norm( int num_inputs, int num_outputs, float *weights, float max_norm ) {
  int i, j, offset;
  float row_total, row_factor;
  float mn_squared = max_norm * max_norm;

  for ( j = 0; j < num_outputs; j++ ) {
    offset = ( num_inputs + 1 ) * j;
    row_total = 0.0;
    for ( i = 0; i < num_inputs; i++ ) {
      row_total += weights[ offset + i ] * weights[ offset + i ];
    }
    if ( row_total > mn_squared ) {
      row_factor = max_norm / sqrt( row_total );
      for ( i = 0; i < num_inputs; i++ ) {
        weights[ offset + i ] *= row_factor;
      }
    }
  }
  return;
}
