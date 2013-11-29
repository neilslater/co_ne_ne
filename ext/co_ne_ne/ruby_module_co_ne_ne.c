// ext/co_ne_ne/ruby_module_co_ne_ne.c

#include "ruby_module_co_ne_ne.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// To hold the module object
VALUE CoNeNe = Qnil;

/* @overload convolve( signal, kernel )
 * Calculates convolution of an array of floats representing a signal, with a second array representing
 * a kernel. The two parameters must have the same rank. The output has same rank, its size in each dimension d is given by
 *  signal.shape[d] - kernel.shape[d] + 1
 * @param [NArray] signal must be same size or larger than kernel in each dimension
 * @param [NArray] kernel must be same size or smaller than signal in each dimension
 * @return [NArray] result of convolving signal with kernel
 */
static VALUE narray_convolve( VALUE self, VALUE a, VALUE b ) {
  struct NARRAY *na_a, *na_b, *na_c;
  volatile VALUE val_a, val_b, val_c;
  int target_rank, i;
  int target_shape[LARGEST_RANK];

  val_a = na_cast_object(a, NA_SFLOAT);
  GetNArray( val_a, na_a );

  val_b = na_cast_object(b, NA_SFLOAT);
  GetNArray( val_b, na_b );

  if ( na_a->rank != na_b->rank ) {
    rb_raise( rb_eArgError, "narray a must have equal rank to narray b (a rack %d, b rank %d)", na_a->rank,  na_b->rank );
  }

  if ( na_a->rank > LARGEST_RANK ) {
    rb_raise( rb_eArgError, "exceeded maximum narray rank for convolve of %d", LARGEST_RANK );
  }

  target_rank = na_a->rank;

  for ( i = 0; i < target_rank; i++ ) {
    target_shape[i] = na_a->shape[i] - na_b->shape[i] + 1;
    if ( target_shape[i] < 1 ) {
      rb_raise( rb_eArgError, "narray b is bigger in one or more dimensions than narray a" );
    }
  }

  val_c = na_make_object( NA_SFLOAT, target_rank, target_shape, CLASS_OF( val_a ) );
  GetNArray( val_c, na_c );

  core_convole(
    target_rank, na_a->shape, (float*) na_a->ptr,
    target_rank, na_b->shape, (float*) na_b->ptr,
    target_rank, target_shape, (float*) na_c->ptr );

  return val_c;
}

/* @overload max_pool( array, tile_size, pool_size )
 * Reduces an array in each dimension by a factor tile_size, by sampling pool_size entries
 * and using the maximum value found.
 * @param [NArray] array source data for pooling
 * @param [Integer] tile_size reduce dimensions of input array by this factor, accepts 1 to 100
 * @param [Integer] pool_size consider these many positions in each dimension (allows for overlap), accepts 1 to 100
 * @return [NArray] result of applying max pooling to array
 */
static VALUE narray_max_pool( VALUE self, VALUE a, VALUE tile_size, VALUE pool_size ) {
  struct NARRAY *na_a, *na_b;
  volatile VALUE val_a, val_b;
  int target_rank, i, tile, pool;
  int target_shape[LARGEST_RANK];

  tile = NUM2INT( tile_size );
  if ( tile < 1 || tile > 100 ) {
    rb_raise( rb_eArgError, "tile size out of bounds, expected in range 1..100, got %d", tile );
  }

  pool = NUM2INT( pool_size );
  if ( pool < 1 || pool > 100 ) {
    rb_raise( rb_eArgError, "pool size out of bounds, expected in range 1..100, got %d", pool );
  }

  val_a = na_cast_object(a, NA_SFLOAT);
  GetNArray( val_a, na_a );

  if ( na_a->rank > LARGEST_RANK ) {
    rb_raise( rb_eArgError, "exceeded maximum narray rank for max_pool of %d", LARGEST_RANK );
  }

  target_rank = na_a->rank;

  for ( i = 0; i < target_rank; i++ ) {
    target_shape[i] = ( na_a->shape[i] + tile - 1 ) / tile;
  }

  val_b = na_make_object( NA_SFLOAT, target_rank, target_shape, CLASS_OF( val_a ) );
  GetNArray( val_b, na_b );

  core_max_pool(
    target_rank, na_a->shape, (float*) na_a->ptr,
    target_shape, (float*) na_b->ptr,
    tile, pool );

  return val_b;
}

/* @overload srand( seed )
 * Seed the random number generator used for weights.
 * @param [Integer] seed 64-bit seed number
 * @return [nil]
 */
static VALUE mt_srand( VALUE self, VALUE seed ) {
  init_genrand( NUM2ULONG( seed ) );
  return Qnil;
}


/* @overload rand( )
 * @!visibility private
 * Use the random number generator (Ruby binding only used for tests)
 * @return [Float] random number in range 0.0..1.0
 */
static VALUE mt_rand_float( VALUE self ) {
  return FLT2NUM( genrand_real1() );
}

void init_module_co_ne_ne() {
  CoNeNe = rb_define_module( "CoNeNe" );
  rb_define_singleton_method( CoNeNe, "convolve", narray_convolve, 2 );
  rb_define_singleton_method( CoNeNe, "max_pool", narray_max_pool, 3 );
  rb_define_singleton_method( CoNeNe, "srand", mt_srand, 1 );
  rb_define_singleton_method( CoNeNe, "rand", mt_rand_float, 0 );
  init_transfer_module();
  init_mlp_layer_class();
  init_mlp_network_class();
  init_srand_by_time();
}
