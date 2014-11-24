// ext/con_ne_ne/struct_layer_ff.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating Layer_FF structs
//

#ifndef STRUCT_MLP_LAYER_H
#define STRUCT_MLP_LAYER_H

#include <ruby.h>
#include "narray.h"
#include "mt.h"
#include "ruby_module_transfer.h"
#include "core_backprop.h"

typedef struct _layer_ff_raw {
    int num_inputs;
    int num_outputs;
    int locked_input;
    transfer_type transfer_fn;
    VALUE narr_input;
    VALUE narr_output;
    VALUE narr_weights;
    VALUE input_layer;
    VALUE output_layer;
    VALUE narr_output_deltas;
    VALUE narr_weights_last_deltas;
    VALUE narr_output_slope;
  } Layer_FF;

Layer_FF *layer_ff__create();

void layer_ff__destroy( Layer_FF *layer_ff );

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void layer_ff__gc_mark( Layer_FF *layer_ff );

void layer_ff__new_narrays( Layer_FF *layer_ff );

void layer_ff__init_weights( Layer_FF *layer_ff, float min, float max );

void layer_ff__init_from_weights( Layer_FF *layer_ff, VALUE weights );

void layer_ff__run( Layer_FF *layer_ff );

void layer_ff__backprop_deltas( Layer_FF *layer_ff, Layer_FF *layer_ff_input );

void layer_ff__update_weights( Layer_FF *layer_ff, float eta, float m );

void layer_ff__calc_output_deltas( Layer_FF *layer_ff, VALUE target );

void layer_ff__set_input( Layer_FF *layer_ff, VALUE val_input );

void layer_ff__clear_input( Layer_FF *layer_ff );

#endif
