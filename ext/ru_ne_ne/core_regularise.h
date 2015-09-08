// ext/ru_ne_ne/core_regularise.h

#ifndef CORE_REGULARISE_H
#define CORE_REGULARISE_H

#include <ruby.h>
#include <xmmintrin.h>
#include "core_narray.h"

void apply_weight_decay( int num_inputs, int num_outputs, float *weights, float *de_dw, float weight_decay );

void apply_max_norm( int num_inputs, int num_outputs, float *weights, float max_norm );

#endif
