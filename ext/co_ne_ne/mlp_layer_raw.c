// ext/co_ne_ne/mlp_layer_raw.c

#include "mlp_layer_raw.h"
#include "narray_shared.h"
#include <xmmintrin.h>

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
  mlp_layer->narr_output_slope = Qnil;

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

  return;
}

// Note this isn't called from initialize_copy, it's for internal copies
// Also note - it is incomplete., and unused.
MLP_Layer *copy_mlp_layer_struct( MLP_Layer *orig ) {
  MLP_Layer *mlp_layer = create_mlp_layer_struct();

  mlp_layer->num_inputs = orig->num_inputs;
  mlp_layer->num_outputs = orig->num_outputs;
  mlp_layer->transfer_fn = orig->transfer_fn;

  // TODO: Clone whatever needs cloning . . .

  return mlp_layer;
}

void mlp_layer_struct_use_weights( MLP_Layer *mlp_layer, VALUE weights ) {
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

void mlp_layer_run( MLP_Layer *mlp_layer ) {
  struct NARRAY *na_in;
  struct NARRAY *na_weights;
  struct NARRAY *na_out;

  GetNArray( mlp_layer->narr_input, na_in );
  GetNArray( mlp_layer->narr_weights, na_weights );
  GetNArray( mlp_layer->narr_output, na_out );

  activate_nn_layer_raw( mlp_layer->num_inputs, mlp_layer->num_outputs,
      (float*) na_in->ptr, (float*) na_weights->ptr, (float*) na_out->ptr );

  transfer_bulk_apply_function( mlp_layer->transfer_fn, mlp_layer->num_outputs, (float*) na_out->ptr  );

  return;
}

void activate_nn_layer_raw( int in_size, int out_size,
    float *in_ptr, float *weights, float *out_ptr ) {
  int i, j, in_aligned_size, offset;
  __m128 simd_x, simd_y, simd_t;
  float v[4];

  in_aligned_size = 4 * ( in_size/4 );

  // Calculate activation
  for ( i = 0; i < out_size; i++ ) {

    float t = 0.0;
    simd_t = _mm_setzero_ps();
    offset = i * (in_size + 1);

    // Use SIMD for all the aligned values in groups of 4
    for ( j = 0; j < in_aligned_size; j +=4 ) {
      simd_x = _mm_load_ps( in_ptr + j );
      // Weights might not align to 16 bytes due to size of layers
      simd_y = _mm_loadu_ps( weights + (offset + j) );
      simd_x = _mm_mul_ps( simd_x, simd_y );
      simd_t = _mm_add_ps( simd_x, simd_t );
    }
    _mm_store_ps( v, simd_t );

    // Complete any remaining 1,2 or 3 items one at a time
    for ( j = in_aligned_size; j < in_size; j++ ) {
      t += in_ptr[ j ] * weights[ offset + j ];
    }

    // Add together 4 simd channels, plus bias
    out_ptr[i] = v[0] + v[1] + v[2] + v[3] + t + weights[ offset + in_size ];
  }

  return;
}

float ms_error_raw( int out_size, float *out_ptr, float *target_ptr ) {
  int i;
  float t = 0.0;
  float d;

  for ( i = 0; i < out_size; i++ ) {
    d = out_ptr[i] - target_ptr[i];
    t += d * d;
  }

  return t/out_size;
}

void calc_output_deltas_raw( int out_size, float *out_ptr, float *out_slope_ptr,
      float *target_ptr, float *out_delta_ptr ) {
  int i;
  for ( i = 0; i < out_size; i++ ) {
    out_delta_ptr[i] = ( target_ptr[i] - out_ptr[i] ) * out_slope_ptr[i];
  }
  return;
}

void backprop_deltas_raw( int in_size, int out_size,
      float *in_deltas, float *in_slopes,
      float *weights, float *out_deltas ) {
  int i,j;
  float t;
  for( i = 0; i < in_size; i++ ) {
    t = 0.0;
    for( j = 0; j < out_size; j++ ) {
      t += weights[ j * (in_size+1) + i ] * out_deltas[j];
    }
    in_deltas[i] = t * in_slopes[i];
  }
  return;
}

void mlp_layer_backprop( MLP_Layer *mlp_layer, MLP_Layer *mlp_layer_input ) {
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

  backprop_deltas_raw( mlp_layer->num_inputs, mlp_layer->num_outputs,
        (float *) na_in_deltas->ptr, (float *) na_in_slopes->ptr,
        (float *) na_weights->ptr, (float *) na_out_deltas->ptr );
}