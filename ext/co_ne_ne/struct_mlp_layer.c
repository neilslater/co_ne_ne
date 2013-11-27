// ext/co_ne_ne/struct_mlp_layer.c

#include "struct_mlp_layer.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating MLP_Layer structs
//

MLP_Layer *p_mlp_layer_create() {
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
  mlp_layer->narr_output_slope = Qnil;
  mlp_layer->locked_input = 0;

  return mlp_layer;
}

// Creates weights, outputs etc
void p_mlp_layer_new_narrays( MLP_Layer *mlp_layer ) {
  int shape[2];
  struct NARRAY *narr;

  shape[0] = mlp_layer->num_outputs;
  mlp_layer->narr_output = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( mlp_layer->narr_output, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  mlp_layer->narr_output_deltas = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( mlp_layer->narr_output_deltas, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  mlp_layer->narr_output_slope = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( mlp_layer->narr_output_slope, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  shape[0] = mlp_layer->num_inputs + 1;
  shape[1] = mlp_layer->num_outputs;
  mlp_layer->narr_weights = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( mlp_layer->narr_weights, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  mlp_layer->narr_weights_last_deltas = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( mlp_layer->narr_weights_last_deltas, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  return;
}

// Creates weights, outputs etc
void p_mlp_layer_init_weights( MLP_Layer *mlp_layer, float min, float max ) {
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

void p_mlp_layer_destroy( MLP_Layer *mlp_layer ) {
  xfree( mlp_layer );
  // No need to free NArrays - they will be handled by Ruby's GC, and may still be reachable
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void p_mlp_layer_gc_mark( MLP_Layer *mlp_layer ) {
  rb_gc_mark( mlp_layer->narr_input );
  rb_gc_mark( mlp_layer->narr_output );
  rb_gc_mark( mlp_layer->narr_weights );
  rb_gc_mark( mlp_layer->input_layer );
  rb_gc_mark( mlp_layer->output_layer );
  rb_gc_mark( mlp_layer->narr_output_deltas );
  rb_gc_mark( mlp_layer->narr_weights_last_deltas );
  rb_gc_mark( mlp_layer->narr_output_slope );

  return;
}

void p_mlp_layer_init_from_weights( MLP_Layer *mlp_layer, VALUE weights ) {
  int shape[2];

  shape[0] = mlp_layer->num_outputs;
  mlp_layer->narr_output = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  mlp_layer->narr_output_deltas = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  mlp_layer->narr_output_slope = na_make_object( NA_SFLOAT, 1, shape, cNArray );

  shape[0] = mlp_layer->num_inputs + 1;
  shape[1] = mlp_layer->num_outputs;
  mlp_layer->narr_weights = weights;
  mlp_layer->narr_weights_last_deltas = na_make_object( NA_SFLOAT, 2, shape, cNArray );

  return;
}

void p_mlp_layer_run( MLP_Layer *mlp_layer ) {
  struct NARRAY *na_in;
  struct NARRAY *na_weights;
  struct NARRAY *na_out;

  GetNArray( mlp_layer->narr_input, na_in );
  GetNArray( mlp_layer->narr_weights, na_weights );
  GetNArray( mlp_layer->narr_output, na_out );

  core_activate_layer_output( mlp_layer->num_inputs, mlp_layer->num_outputs,
      (float*) na_in->ptr, (float*) na_weights->ptr, (float*) na_out->ptr );

  transfer_bulk_apply_function( mlp_layer->transfer_fn, mlp_layer->num_outputs, (float*) na_out->ptr  );

  return;
}

void p_mlp_layer_backprop_deltas( MLP_Layer *mlp_layer, MLP_Layer *mlp_layer_input ) {
  struct NARRAY *na_inputs; // Only required to pre-calc slopes

  struct NARRAY *na_in_deltas;
  struct NARRAY *na_in_slopes;
  struct NARRAY *na_weights;
  struct NARRAY *na_out_deltas;

  GetNArray( mlp_layer_input->narr_output, na_inputs );

  GetNArray( mlp_layer_input->narr_output_deltas, na_in_deltas );
  GetNArray( mlp_layer_input->narr_output_slope, na_in_slopes );

  GetNArray( mlp_layer->narr_weights, na_weights );

  GetNArray( mlp_layer->narr_output_deltas, na_out_deltas );

  transfer_bulk_derivative_at( mlp_layer_input->transfer_fn, mlp_layer_input->num_outputs,
         (float *) na_inputs->ptr, (float *) na_in_slopes->ptr );

  core_backprop_deltas( mlp_layer->num_inputs, mlp_layer->num_outputs,
        (float *) na_in_deltas->ptr, (float *) na_in_slopes->ptr,
        (float *) na_weights->ptr, (float *) na_out_deltas->ptr );
}

void p_mlp_layer_update_weights( MLP_Layer *mlp_layer, float eta, float m ) {
  struct NARRAY *na_input;
  struct NARRAY *na_weights;
  struct NARRAY *na_weights_last_deltas;
  struct NARRAY *na_output_deltas;

  GetNArray( mlp_layer->narr_input, na_input );
  GetNArray( mlp_layer->narr_weights, na_weights );
  GetNArray( mlp_layer->narr_weights_last_deltas, na_weights_last_deltas );
  GetNArray( mlp_layer->narr_output_deltas, na_output_deltas );

  core_update_weights( eta, m, mlp_layer->num_inputs, mlp_layer->num_outputs,
        (float *) na_input->ptr, (float *) na_weights->ptr,
        (float *) na_weights_last_deltas->ptr, (float *) na_output_deltas->ptr );
}

void p_mlp_layer_calc_output_deltas( MLP_Layer *mlp_layer, VALUE target ) {
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  struct NARRAY *na_output_slope;
  struct NARRAY *na_output_deltas;

  GetNArray( target, na_target );
  GetNArray( mlp_layer->narr_output, na_output );
  GetNArray( mlp_layer->narr_output_slope, na_output_slope );
  GetNArray( mlp_layer->narr_output_deltas, na_output_deltas );

  transfer_bulk_derivative_at( mlp_layer->transfer_fn, mlp_layer->num_outputs,
      (float *) na_output->ptr, (float *) na_output_slope->ptr );

  core_calc_output_deltas( mlp_layer->num_outputs, (float *) na_output->ptr,
      (float *) na_output_slope->ptr, (float *) na_target->ptr, (float *) na_output_deltas->ptr );

  return;
}

void p_mlp_layer_set_input( MLP_Layer *mlp_layer, VALUE val_input ) {
  MLP_Layer *mlp_old_input_layer;

  if ( ! NIL_P( mlp_layer->input_layer ) ) {
    // This layer has an existing input layer, it needs to stop pointing its output here
    Data_Get_Struct( mlp_layer->input_layer, MLP_Layer, mlp_old_input_layer );
    mlp_old_input_layer->output_layer = Qnil;
  }

  mlp_layer->narr_input = val_input;
  mlp_layer->input_layer = Qnil;

  return;
}

void p_mlp_layer_clear_input( MLP_Layer *mlp_layer ) {
  MLP_Layer *mlp_old_input_layer;

  if ( ! NIL_P( mlp_layer->input_layer ) ) {
    // This layer has an existing input layer, it needs to stop pointing its output here
    Data_Get_Struct( mlp_layer->input_layer, MLP_Layer, mlp_old_input_layer );
    mlp_old_input_layer->output_layer = Qnil;
  }

  mlp_layer->narr_input = Qnil;
  mlp_layer->input_layer = Qnil;

  return;
}
