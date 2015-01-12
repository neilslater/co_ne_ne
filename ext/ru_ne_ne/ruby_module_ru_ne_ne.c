// ext/ru_ne_ne/ruby_module_ru_ne_ne.c

#include "ruby_module_ru_ne_ne.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// To hold the module object, plus child Module and Class items
VALUE RuNeNe = Qnil;

VALUE RuNeNe_Transfer = Qnil;
VALUE RuNeNe_Transfer_Sigmoid = Qnil;
VALUE RuNeNe_Transfer_TanH = Qnil;
VALUE RuNeNe_Transfer_ReLU = Qnil;
VALUE RuNeNe_Transfer_Linear = Qnil;
VALUE RuNeNe_Transfer_Softmax = Qnil;

VALUE RuNeNe_Layer = Qnil;
VALUE RuNeNe_Layer_FeedForward  = Qnil;

VALUE RuNeNe_Network = Qnil;

VALUE RuNeNe_TrainingData = Qnil;

/* @overload convolve( signal, kernel )
 * Calculates convolution of an array of floats representing a signal, with a second array representing
 * a kernel. The two parameters must have the same rank. The output has same rank, its size in each dimension d is given by
 *  signal.shape[d] - kernel.shape[d] + 1
 * @param [NArray] signal must be same size or larger than kernel in each dimension
 * @param [NArray] kernel must be same size or smaller than signal in each dimension
 * @return [NArray] result of convolving signal with kernel
 */
static VALUE narray_convolve( VALUE self, VALUE rv_a, VALUE rv_b ) {
  struct NARRAY *na_a, *na_b, *na_c;
  volatile VALUE val_a, val_b, val_c;
  int target_rank, i;
  int target_shape[LARGEST_RANK];

  val_a = na_cast_object( rv_a, NA_SFLOAT );
  GetNArray( val_a, na_a );

  val_b = na_cast_object( rv_b, NA_SFLOAT );
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
static VALUE narray_max_pool( VALUE self, VALUE rv_a, VALUE rv_tile_size, VALUE rv_pool_size ) {
  struct NARRAY *na_a, *na_b;
  volatile VALUE val_a, val_b;
  int target_rank, i, tile, pool;
  int target_shape[LARGEST_RANK];

  tile = NUM2INT( rv_tile_size );
  if ( tile < 1 || tile > 100 ) {
    rb_raise( rb_eArgError, "tile size out of bounds, expected in range 1..100, got %d", tile );
  }

  pool = NUM2INT( rv_pool_size );
  if ( pool < 1 || pool > 100 ) {
    rb_raise( rb_eArgError, "pool size out of bounds, expected in range 1..100, got %d", pool );
  }

  val_a = na_cast_object( rv_a, NA_SFLOAT);
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
 * @param [Integer] seed 32-bit seed number
 * @return [nil]
 */
static VALUE mt_srand( VALUE self, VALUE rv_seed ) {
  init_genrand( NUM2ULONG( rv_seed ) );
  return Qnil;
}

static unsigned long runene_srand_seed[640];

/* @overload srand_array( seed )
 * Seed the random number generator used for weights.
 * @param [Array<Integer>] seed an array of up to 640 times 32 bit seed numbers
 * @return [nil]
 */
static VALUE mt_srand_array( VALUE self, VALUE rv_seed_array ) {
  int i, n;
  Check_Type( rv_seed_array, T_ARRAY );
  n = FIX2INT( rb_funcall( rv_seed_array, rb_intern("count"), 0 ) );
  if ( n < 1 ) {
    rb_raise( rb_eArgError, "empty array cannot be used to seed RNG" );
  }
  if ( n > 640 ) { n = 640; }
  for ( i = 0; i < n; i++ ) {
    runene_srand_seed[i] = NUM2ULONG( rb_ary_entry( rv_seed_array, i ) );
  }
  init_by_array( runene_srand_seed, n );
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

/* @overload shuffled_integers( n )
 * @!visibility private
 * Uses internal sort and RNG to
 * @param [Integer] n size of array to shuffle
 * @return [Array] numbers 0...n in random order
 */
static VALUE runene_shuffled_integers( VALUE self, VALUE rv_n ) {
  int n, *ids, i;
  volatile VALUE arr;
  n = NUM2INT( rv_n );
  if ( n < 1 ) {
    rb_raise( rb_eArgError, "number of integers to shuffle must be 1 or more, got %d", n );
  }

  ids = ALLOC_N( int, n );
  for ( i = 0; i < n; i++ ) { ids[i] = i; }
  shuffle_ints( n, ids );

  arr = rb_ary_new2( n );
  for ( i = 0; i < n; i++ ) {
    rb_ary_store( arr, i, INT2NUM(ids[i]) );
  }

  xfree( ids );
  return arr;
}

void init_module_ru_ne_ne() {
  RuNeNe = rb_define_module( "RuNeNe" );

  RuNeNe_Transfer = rb_define_module_under( RuNeNe, "Transfer" );
  RuNeNe_Transfer_Sigmoid = rb_define_module_under( RuNeNe_Transfer, "Sigmoid" );
  RuNeNe_Transfer_TanH = rb_define_module_under( RuNeNe_Transfer, "TanH" );
  RuNeNe_Transfer_ReLU = rb_define_module_under( RuNeNe_Transfer, "ReLU" );
  RuNeNe_Transfer_Linear = rb_define_module_under( RuNeNe_Transfer, "Linear" );
  RuNeNe_Transfer_Softmax = rb_define_module_under( RuNeNe_Transfer, "Softmax" );

  RuNeNe_Layer = rb_define_class_under( RuNeNe, "Layer", rb_cObject );
  RuNeNe_Layer_FeedForward = rb_define_class_under( RuNeNe_Layer, "FeedForward", rb_cObject );

  RuNeNe_Network = rb_define_class_under( RuNeNe, "Network", rb_cObject );

  RuNeNe_TrainingData = rb_define_class_under( RuNeNe, "TrainingData", rb_cObject );

  rb_define_singleton_method( RuNeNe, "convolve", narray_convolve, 2 );
  rb_define_singleton_method( RuNeNe, "max_pool", narray_max_pool, 3 );
  rb_define_singleton_method( RuNeNe, "srand", mt_srand, 1 );
  rb_define_singleton_method( RuNeNe, "srand_array", mt_srand_array, 1 );
  rb_define_singleton_method( RuNeNe, "rand", mt_rand_float, 0 );
  rb_define_singleton_method( RuNeNe, "shuffled_integers", runene_shuffled_integers, 1 );

  init_transfer_module();
  init_layer_ff_class();
  init_network_class();
  init_training_data_class();
  init_srand_by_time();
}
