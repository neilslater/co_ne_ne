// ext/co_ne_ne/mlp_layer_raw.c

#include "mlp_layer_raw.h"

MLP_Layer *create_mlp_layer_struct() {
  MLP_Layer *mlp_layer;
  mlp_layer = xmalloc( sizeof(MLP_Layer) );
  mlp_layer->num_inputs = 0;
  mlp_layer->num_outputs = 0;
  mlp_layer->transfer_fn_id = 0; // sigmoid
  mlp_layer->narr_input = Qnil;
  mlp_layer->narr_output = Qnil;
  mlp_layer->narr_weights = Qnil;
  mlp_layer->input_layer = Qnil;
  mlp_layer->output_layer = Qnil;
  mlp_layer->narr_output_deltas = Qnil;
  mlp_layer->weights_last_deltas = Qnil;
  return mlp_layer;
}

void destroy_mlp_layer_struct( MLP_Layer *mlp_layer ) {
  xfree( mlp_layer );
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void mark_mlp_layer_struct( MLP_Layer *mlp_layer ) {

  return;
}

// Note this isn't called from initialize_copy, it's for internal copies
MLP_Layer *copy_mlp_layer_struct( MLP_Layer *orig ) {
  MLP_Layer *mlp_layer = create_mlp_layer_struct();

  mlp_layer->num_inputs = orig->num_inputs;
  mlp_layer->num_outputs = orig->num_outputs;
  mlp_layer->transfer_fn_id = orig->transfer_fn_id;

  // Clone now, or later?
  mlp_layer->narr_input = Qnil;
  mlp_layer->narr_output = Qnil;
  mlp_layer->narr_weights = Qnil;
  mlp_layer->input_layer = Qnil;
  mlp_layer->output_layer = Qnil;
  mlp_layer->narr_output_deltas = Qnil;
  mlp_layer->weights_last_deltas = Qnil;

  return mlp_layer;
}
