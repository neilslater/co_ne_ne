// ext/con_ne_ne/struct_mlp_layer.h

#ifndef STRUCT_MLP_LAYER_H
#define STRUCT_MLP_LAYER_H

#include <ruby.h>
#include "narray.h"
#include "core_mt.h"
#include "ruby_module_transfer.h"
#include "core_narray.h"
#include <xmmintrin.h>
#include "core_backprop.h"

typedef struct _mlp_layer_raw {
    int num_inputs;
    int num_outputs;
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

MLP_Layer *create_mlp_layer_struct();

void destroy_mlp_layer_struct( MLP_Layer *mlp_layer );

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void mark_mlp_layer_struct( MLP_Layer *mlp_layer );

// Note this isn't called from initialize_copy, it's for internal copies
MLP_Layer *copy_mlp_layer_struct( MLP_Layer *orig );

void mlp_layer_struct_create_arrays( MLP_Layer *mlp_layer );

void mlp_layer_struct_init_weights( MLP_Layer *mlp_layer, float min, float max );

void mlp_layer_struct_use_weights( MLP_Layer *mlp_layer, VALUE weights );

void mlp_layer_run( MLP_Layer *mlp_layer );

void mlp_layer_backprop( MLP_Layer *mlp_layer, MLP_Layer *mlp_layer_input );

void mlp_layer_struct_update_weights( MLP_Layer *mlp_layer, float eta, float m );

#endif
