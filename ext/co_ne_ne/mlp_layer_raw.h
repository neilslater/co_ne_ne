// ext/con_ne_ne/mlp_layer_raw.h

#ifndef MLP_LAYER_RAW_H
#define MLP_LAYER_RAW_H

#include <ruby.h>
#include "narray.h"

typedef struct _mlp_layer_raw {
    int num_inputs;
    int num_outputs;
    int transfer_fn_id;
    VALUE narr_input;
    VALUE narr_output;
    VALUE narr_weights;
    VALUE input_layer;
    VALUE output_layer;
    VALUE narr_output_deltas;
    VALUE weights_last_deltas;
  } MLP_Layer;

MLP_Layer *create_mlp_layer_struct();

void destroy_mlp_layer_struct( MLP_Layer *mlp_layer );

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void mark_mlp_layer_struct( MLP_Layer *mlp_layer );

// Note this isn't called from initialize_copy, it's for internal copies
MLP_Layer *copy_mlp_layer_struct( MLP_Layer *orig );

#endif
