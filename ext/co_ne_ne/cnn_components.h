// ext/co_ne_ne/cnn_components.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of narray helper functions
//

#ifndef CNN_COMPONENTS_H
#define CNN_COMPONENTS_H

void max_pool_raw( int rank, int *input_shape, float *input_ptr,
    int *output_shape, float *output_ptr, int tile_by, int pool_by );

#endif
