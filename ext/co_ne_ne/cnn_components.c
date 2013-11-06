// ext/co_ne_ne/cnn_components.c

#include <xmmintrin.h>
#include "cnn_components.h"

// This is copied from na_array.c, with safety checks and temp vars removed
inline int cnnc_inline_idxs_to_pos( int rank, int *shape, int *idxs ) {
  int i, pos = 0;
  for ( i = rank - 1; i >= 0; i-- ) {
    if ( idxs[i] >= shape[i] ) {
      pos = -1;
      break;
    }
    pos = pos * shape[i] + idxs[i];
  }
  return pos;
}

// Starts indices
inline void indices_reset( int rank, int *indices ) {
  int i;
  for ( i = 0; i < rank; i++ ) { indices[i] = 0; }
  return;
}

// Increments indices
inline int indices_inc( int rank, int *shape, int *indices ) {
  int i = 0;
  while ( indices[i]++ > shape[i] - 2 ) {
    indices[i] = 0;
    i++;
  }
  return i;
}


inline int size_from_shape2( int rank, int *shape ) {
  int size = 1;
  int i;
  for ( i = 0; i < rank; i++ ) { size *= shape[i]; }
  return size;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Max-pool multi-dimension array
//
//    Benchmark: 256x256 inputs, tile 2, pool 3. 1000 iterations. 1.68 seconds.
//
//

void max_pool_raw( int rank, int *input_shape, float *input_ptr,
    int *output_shape, float *output_ptr, int tile_by, int pool_by ) {
  int i, j, k, pool_size, output_size, pos;
  int output_idx[16], input_idx[16], pool_idx[16], pool_shape[16];
  double max;

  for ( i = 0; i < rank; i++ ) { pool_shape[i] = pool_by; }
  pool_size = size_from_shape2( rank, pool_shape );
  output_size = size_from_shape2( rank, output_shape );

  indices_reset( rank, output_idx );
  for (i = 0; i < output_size; i++ ) {
    max = -1e30;
    indices_reset( rank, pool_idx );
    for (j = 0; j < pool_size; j++ ) {
      for ( k = 0; k < rank; k++ ) { input_idx[k] = output_idx[k] * tile_by + pool_idx[k]; }
      pos = cnnc_inline_idxs_to_pos( rank, input_shape, input_idx );
      if ( pos >= 0 && input_ptr[pos] > max ) {
        max = input_ptr[pos];
      }
      indices_inc( rank, pool_shape, pool_idx );
    }
    output_ptr[i] = max;
    indices_inc( rank, output_shape, output_idx );
  }

  return;
}
