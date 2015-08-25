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
  layer_ff->narr_weights = Qnil;
  layer_ff->weights = NULL;

  return layer_ff;
}

// Creates weights, outputs etc
void layer_ff__new_narrays( Layer_FF *layer_ff ) {
  int shape[2];
  struct NARRAY *narr;

  shape[0] = layer_ff->num_inputs + 1;
  shape[1] = layer_ff->num_outputs;
  layer_ff->narr_weights = na_make_object( NA_SFLOAT, 2, shape, cNArray );
  GetNArray( layer_ff->narr_weights, narr );
  layer_ff->weights = (float*) narr->ptr;
  na_sfloat_set( narr->total, layer_ff->weights, (float) 0.0 );

  return;
}

// Creates weights, using randn()
void layer_ff__init_weights( Layer_FF *layer_ff ) {
  int i;

  struct NARRAY *narr;
  GetNArray( layer_ff->narr_weights, narr );
  int t = narr->total;

  double sigma = 0.5 * sqrt ( 6.0 / ( layer_ff->num_inputs + layer_ff->num_outputs ));
  for ( i = 0; i < t; i++ ) {
    layer_ff->weights[i] = sigma * genrand_norm();
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
  rb_gc_mark( layer_ff->narr_weights );
  return;
}

void layer_ff__set_weights( Layer_FF *layer_ff, VALUE weights ) {
  struct NARRAY *narr;
  layer_ff->narr_weights = weights;
  GetNArray( layer_ff->narr_weights, narr );
  layer_ff->weights = (float*) narr->ptr;
  return;
}

void feed_forward_linear( int in_size, int out_size, float *in_ptr, float *weights, float *out_ptr ) {
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
      // Unfortunately loadu is required, we just don't know the offset
      simd_x = _mm_loadu_ps( in_ptr + j );
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

void layer_ff__run( Layer_FF *layer_ff, float *input, float *output ) {
  feed_forward_linear( layer_ff->num_inputs, layer_ff->num_outputs, input, layer_ff->weights, output );
  transfer_bulk_apply_function( layer_ff->transfer_fn, layer_ff->num_outputs, output );
  return;
}
