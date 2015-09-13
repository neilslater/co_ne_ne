// ext/ru_ne_ne/struct_layer_ff.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating Layer_FF structs
//

#ifndef STRUCT_LAYER_FF_H
#define STRUCT_LAYER_FF_H

#include <ruby.h>
#include "narray.h"
#include "mt.h"
#include "core_narray.h"
#include <xmmintrin.h>

#include "ruby_module_transfer.h"

typedef struct _layer_ff_raw {
    int num_inputs;
    int num_outputs;
    transfer_type transfer_fn;
    volatile VALUE narr_weights;
    float * weights;
  } Layer_FF;

Layer_FF *layer_ff__create();

void layer_ff__destroy( Layer_FF *layer_ff );

void layer_ff__gc_mark( Layer_FF *layer_ff );

void layer_ff__new_narrays( Layer_FF *layer_ff );

void layer_ff__init_weights( Layer_FF *layer_ff );

void layer_ff__set_weights( Layer_FF *layer_ff, VALUE weights );

void layer_ff__run( Layer_FF *layer_ff, float *input, float *output );

#endif
