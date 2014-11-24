// ext/ru_ne_ne/struct_layer_ff.c

#include "struct_layer_ff.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating Layer_FF structs
//

Layer_FF *layer_ff__create() {
  Layer_FF *layer_ff;
  layer_ff = xmalloc( sizeof(Layer_FF) );
  layer_ff->num_inputs = 0;
  layer_ff->num_outputs = 0;
  layer_ff->transfer_fn = SIGMOID;
  layer_ff->narr_input = Qnil;
  layer_ff->narr_output = Qnil;
  layer_ff->narr_weights = Qnil;
  layer_ff->input_layer = Qnil;
  layer_ff->output_layer = Qnil;
  layer_ff->narr_output_deltas = Qnil;
  layer_ff->narr_weights_last_deltas = Qnil;
  layer_ff->narr_output_slope = Qnil;
  layer_ff->locked_input = 0;

  return layer_ff;
}

// Creates weights, outputs etc
void layer_ff__new_narrays( Layer_FF *layer_ff ) {
  int shape[2];
  struct NARRAY *narr;

  shape[0] = layer_ff->num_outputs;
  layer_ff->narr_output = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( layer_ff->narr_output, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  layer_ff->narr_output_deltas = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( layer_ff->narr_output_deltas, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  layer_ff->narr_output_slope = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( layer_ff->narr_output_slope, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  shape[0] = layer_ff->num_inputs + 1;
  shape[1] = layer_ff->num_outputs;
  layer_ff->narr_weights = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( layer_ff->narr_weights, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  layer_ff->narr_weights_last_deltas = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( layer_ff->narr_weights_last_deltas, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  return;
}

// Creates weights, outputs etc
void layer_ff__init_weights( Layer_FF *layer_ff, float min, float max ) {
  int i, t;
  float mul, *ptr;
  struct NARRAY *narr;

  mul = max - min;
  GetNArray( layer_ff->narr_weights, narr );
  ptr = (float*) narr->ptr;
  t = narr->total;

  for ( i = 0; i < t; i++ ) {
    ptr[i] = min + mul * genrand_real1();
  }

  return;
}

void layer_ff__destroy( Layer_FF *layer_ff ) {
  xfree( layer_ff );
  // No need to free NArrays - they will be handled by Ruby's GC, and may still be reachable
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void layer_ff__gc_mark( Layer_FF *layer_ff ) {
  rb_gc_mark( layer_ff->narr_input );
  rb_gc_mark( layer_ff->narr_output );
  rb_gc_mark( layer_ff->narr_weights );
  rb_gc_mark( layer_ff->input_layer );
  rb_gc_mark( layer_ff->output_layer );
  rb_gc_mark( layer_ff->narr_output_deltas );
  rb_gc_mark( layer_ff->narr_weights_last_deltas );
  rb_gc_mark( layer_ff->narr_output_slope );

  return;
}

void layer_ff__init_from_weights( Layer_FF *layer_ff, VALUE weights ) {
  int shape[2];

  shape[0] = layer_ff->num_outputs;
  layer_ff->narr_output = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  layer_ff->narr_output_deltas = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  layer_ff->narr_output_slope = na_make_object( NA_SFLOAT, 1, shape, cNArray );

  shape[0] = layer_ff->num_inputs + 1;
  shape[1] = layer_ff->num_outputs;
  layer_ff->narr_weights = weights;
  layer_ff->narr_weights_last_deltas = na_make_object( NA_SFLOAT, 2, shape, cNArray );

  return;
}

void layer_ff__run( Layer_FF *layer_ff ) {
  struct NARRAY *na_in;
  struct NARRAY *na_weights;
  struct NARRAY *na_out;

  GetNArray( layer_ff->narr_input, na_in );
  GetNArray( layer_ff->narr_weights, na_weights );
  GetNArray( layer_ff->narr_output, na_out );

  feed_forward_linear( layer_ff->num_inputs, layer_ff->num_outputs,
      (float*) na_in->ptr, (float*) na_weights->ptr, (float*) na_out->ptr );

  transfer_bulk_apply_function( layer_ff->transfer_fn, layer_ff->num_outputs, (float*) na_out->ptr  );

  return;
}

void layer_ff__backprop_deltas( Layer_FF *layer_ff, Layer_FF *layer_ff_input ) {
  struct NARRAY *na_inputs; // Only required to pre-calc slopes

  struct NARRAY *na_in_deltas;
  struct NARRAY *na_in_slopes;
  struct NARRAY *na_weights;
  struct NARRAY *na_out_deltas;

  GetNArray( layer_ff_input->narr_output, na_inputs );

  GetNArray( layer_ff_input->narr_output_deltas, na_in_deltas );
  GetNArray( layer_ff_input->narr_output_slope, na_in_slopes );

  GetNArray( layer_ff->narr_weights, na_weights );

  GetNArray( layer_ff->narr_output_deltas, na_out_deltas );

  transfer_bulk_derivative_at( layer_ff_input->transfer_fn, layer_ff_input->num_outputs,
         (float *) na_inputs->ptr, (float *) na_in_slopes->ptr );

  core_backprop_deltas( layer_ff->num_inputs, layer_ff->num_outputs,
        (float *) na_in_deltas->ptr, (float *) na_in_slopes->ptr,
        (float *) na_weights->ptr, (float *) na_out_deltas->ptr );
}

void layer_ff__update_weights( Layer_FF *layer_ff, float eta, float m ) {
  struct NARRAY *na_input;
  struct NARRAY *na_weights;
  struct NARRAY *na_weights_last_deltas;
  struct NARRAY *na_output_deltas;

  GetNArray( layer_ff->narr_input, na_input );
  GetNArray( layer_ff->narr_weights, na_weights );
  GetNArray( layer_ff->narr_weights_last_deltas, na_weights_last_deltas );
  GetNArray( layer_ff->narr_output_deltas, na_output_deltas );

  core_update_weights( eta, m, layer_ff->num_inputs, layer_ff->num_outputs,
        (float *) na_input->ptr, (float *) na_weights->ptr,
        (float *) na_weights_last_deltas->ptr, (float *) na_output_deltas->ptr );
}

void layer_ff__calc_output_deltas( Layer_FF *layer_ff, VALUE target ) {
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  struct NARRAY *na_output_slope;
  struct NARRAY *na_output_deltas;

  GetNArray( target, na_target );
  GetNArray( layer_ff->narr_output, na_output );
  GetNArray( layer_ff->narr_output_slope, na_output_slope );
  GetNArray( layer_ff->narr_output_deltas, na_output_deltas );

  transfer_bulk_derivative_at( layer_ff->transfer_fn, layer_ff->num_outputs,
      (float *) na_output->ptr, (float *) na_output_slope->ptr );

  core_calc_output_deltas( layer_ff->num_outputs, (float *) na_output->ptr,
      (float *) na_output_slope->ptr, (float *) na_target->ptr, (float *) na_output_deltas->ptr );

  return;
}

void layer_ff__set_input( Layer_FF *layer_ff, VALUE val_input ) {
  Layer_FF *s_old_input_layer_ff;

  if ( ! NIL_P( layer_ff->input_layer ) ) {
    // This layer has an existing input layer, it needs to stop pointing its output here
    Data_Get_Struct( layer_ff->input_layer, Layer_FF, s_old_input_layer_ff );
    s_old_input_layer_ff->output_layer = Qnil;
  }

  layer_ff->narr_input = val_input;
  layer_ff->input_layer = Qnil;

  return;
}

void layer_ff__clear_input( Layer_FF *layer_ff ) {
  Layer_FF *s_old_input_layer_ff;

  if ( ! NIL_P( layer_ff->input_layer ) ) {
    // This layer has an existing input layer, it needs to stop pointing its output here
    Data_Get_Struct( layer_ff->input_layer, Layer_FF, s_old_input_layer_ff );
    s_old_input_layer_ff->output_layer = Qnil;
  }

  layer_ff->narr_input = Qnil;
  layer_ff->input_layer = Qnil;

  return;
}
