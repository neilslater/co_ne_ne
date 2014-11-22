// ext/con_ne_ne/struct_layer_ff.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating s_Layer_FF structs
//

#ifndef STRUCT_MLP_LAYER_H
#define STRUCT_MLP_LAYER_H

#include <ruby.h>
#include "narray.h"
#include "core_mt.h"
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
  } s_Layer_FF;

s_Layer_FF *p_layer_ff_create();

void p_layer_ff_destroy( s_Layer_FF *layer_ff );

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void p_layer_ff_gc_mark( s_Layer_FF *layer_ff );

void p_layer_ff_new_narrays( s_Layer_FF *layer_ff );

void p_layer_ff_init_weights( s_Layer_FF *layer_ff, float min, float max );

void p_layer_ff_init_from_weights( s_Layer_FF *layer_ff, VALUE weights );

void p_layer_ff_run( s_Layer_FF *layer_ff );

void p_layer_ff_backprop_deltas( s_Layer_FF *layer_ff, s_Layer_FF *layer_ff_input );

void p_layer_ff_update_weights( s_Layer_FF *layer_ff, float eta, float m );

void p_layer_ff_calc_output_deltas( s_Layer_FF *layer_ff, VALUE target );

void p_layer_ff_set_input( s_Layer_FF *layer_ff, VALUE val_input );

void p_layer_ff_clear_input( s_Layer_FF *layer_ff );

#endif
