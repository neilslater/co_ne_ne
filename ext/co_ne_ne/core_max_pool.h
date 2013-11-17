// ext/co_ne_ne/core_max_pool.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of narray helper functions
//

#ifndef CORE_MAX_POOL_H
#define CORE_MAX_POOL_H

#include <xmmintrin.h>

void max_pool_raw( int rank, int *input_shape, float *input_ptr,
    int *output_shape, float *output_ptr, int tile_by, int pool_by );

#endif
