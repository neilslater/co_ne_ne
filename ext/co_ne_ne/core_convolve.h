// ext/co_ne_ne/core_convolve.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of convolution functions
//

#ifndef CORE_CONVOLVE_H
#define CORE_CONVOLVE_H

#include <ruby.h>
#include <xmmintrin.h>
#include "core_narray.h"

#define LARGEST_RANK 16

void core_convole(
    int in_rank, int *in_shape, float *in_ptr,
    int kernel_rank, int *kernel_shape, float *kernel_ptr,
    int out_rank, int *out_shape, float *out_ptr );

#endif
