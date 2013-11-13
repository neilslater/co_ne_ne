// ext/co_ne_ne/mlp_layer_raw.c

#include "mlp_layer_raw.h"
#include "narray_shared.h"

MLP_Layer *create_mlp_layer_struct() {
  MLP_Layer *mlp_layer;
  mlp_layer = xmalloc( sizeof(MLP_Layer) );
  mlp_layer->num_inputs = 0;
  mlp_layer->num_outputs = 0;
  mlp_layer->transfer_fn = SIGMOID;
  mlp_layer->narr_input = Qnil;
  mlp_layer->narr_output = Qnil;
  mlp_layer->narr_weights = Qnil;
  mlp_layer->input_layer = Qnil;
  mlp_layer->output_layer = Qnil;
  mlp_layer->narr_output_deltas = Qnil;
  mlp_layer->narr_weights_last_deltas = Qnil;
  // Unused, but may improve performance
  mlp_layer->narr_output_slope = Qnil;
  mlp_layer->narr_input_slope = Qnil;

  return mlp_layer;
}

// Creates weights, outputs etc
void mlp_layer_struct_create_arrays( MLP_Layer *mlp_layer ) {
  int shape[2];

  shape[0] = mlp_layer->num_outputs;
  mlp_layer->narr_output = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  mlp_layer->narr_output_deltas = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  mlp_layer->narr_output_slope = na_make_object( NA_SFLOAT, 1, shape, cNArray );


  shape[0] = mlp_layer->num_inputs + 1;
  shape[1] = mlp_layer->num_outputs;
  mlp_layer->narr_weights = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  mlp_layer->narr_weights_last_deltas = na_make_object( NA_SFLOAT, 2, shape, cNArray );

  return;
}

// Creates weights, outputs etc
void mlp_layer_struct_init_weights( MLP_Layer *mlp_layer, float min, float max ) {
  int i, t;
  float mul, *ptr;
  struct NARRAY *narr;

  mul = max - min;
  GetNArray( mlp_layer->narr_weights, narr );
  ptr = (float*) narr->ptr;
  t = narr->total;

  for ( i = 0; i < t; i++ ) {
    ptr[i] = min + mul * genrand_real1();
  }

  return;
}


void destroy_mlp_layer_struct( MLP_Layer *mlp_layer ) {
  xfree( mlp_layer );
  // No need to free NArrays - they will be handled by Ruby's GC, and may still be reachable
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void mark_mlp_layer_struct( MLP_Layer *mlp_layer ) {
  rb_gc_mark( mlp_layer->narr_input );
  rb_gc_mark( mlp_layer->narr_output );
  rb_gc_mark( mlp_layer->narr_weights );
  rb_gc_mark( mlp_layer->input_layer );
  rb_gc_mark( mlp_layer->output_layer );
  rb_gc_mark( mlp_layer->narr_output_deltas );
  rb_gc_mark( mlp_layer->narr_weights_last_deltas );
  rb_gc_mark( mlp_layer->narr_output_slope );
  rb_gc_mark( mlp_layer->narr_input_slope );

  return;
}

// Note this isn't called from initialize_copy, it's for internal copies
MLP_Layer *copy_mlp_layer_struct( MLP_Layer *orig ) {
  MLP_Layer *mlp_layer = create_mlp_layer_struct();

  mlp_layer->num_inputs = orig->num_inputs;
  mlp_layer->num_outputs = orig->num_outputs;
  mlp_layer->transfer_fn = orig->transfer_fn;

  // Clone now, or later?
  mlp_layer->narr_input = Qnil;
  mlp_layer->narr_output = Qnil;
  mlp_layer->narr_weights = Qnil;
  mlp_layer->input_layer = Qnil;
  mlp_layer->output_layer = Qnil;
  mlp_layer->narr_output_deltas = Qnil;
  mlp_layer->narr_weights_last_deltas = Qnil;

  return mlp_layer;
}
