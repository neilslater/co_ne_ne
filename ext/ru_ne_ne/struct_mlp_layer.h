// ext/con_ne_ne/struct_mlp_layer.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating MLP_Layer structs
//

#ifndef STRUCT_MLP_LAYER_H
#define STRUCT_MLP_LAYER_H

#include <ruby.h>
#include "narray.h"
#include "core_mt.h"
#include "ruby_module_transfer.h"
#include "core_backprop.h"

typedef struct _mlp_layer_raw {
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
  } MLP_Layer;

MLP_Layer *p_mlp_layer_create();

void p_mlp_layer_destroy( MLP_Layer *mlp_layer );

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void p_mlp_layer_gc_mark( MLP_Layer *mlp_layer );

void p_mlp_layer_new_narrays( MLP_Layer *mlp_layer );

void p_mlp_layer_init_weights( MLP_Layer *mlp_layer, float min, float max );

void p_mlp_layer_init_from_weights( MLP_Layer *mlp_layer, VALUE weights );

void p_mlp_layer_run( MLP_Layer *mlp_layer );

void p_mlp_layer_backprop_deltas( MLP_Layer *mlp_layer, MLP_Layer *mlp_layer_input );

void p_mlp_layer_update_weights( MLP_Layer *mlp_layer, float eta, float m );

void p_mlp_layer_calc_output_deltas( MLP_Layer *mlp_layer, VALUE target );

void p_mlp_layer_set_input( MLP_Layer *mlp_layer, VALUE val_input );

void p_mlp_layer_clear_input( MLP_Layer *mlp_layer );

#endif
