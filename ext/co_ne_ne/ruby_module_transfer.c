// ext/co_ne_ne/ruby_module_transfer.c

#include "ruby_module_transfer.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// To hold the module objects
//

VALUE Transfer = Qnil;
VALUE Sigmoid = Qnil;
VALUE TanH = Qnil;
VALUE ReLU = Qnil;
VALUE Linear = Qnil;

/* Document-module:  CoNeNe::Transfer::Sigmoid
 *
 * This is a tried-and-tested transfer function which has desirable properties
 * for backpropagation training. It is used by default in CoNeNe for the last (output)
 * layer. It returns from y =~ 0.0 for large negative x, y = 0.5 when x = 0.0, and  y=~ 1.0
 * for large positive x.
 */

/* @overload function( x )
 * Calculates value of
 *     y = 1.0 / ( 1.0 + exp( -x ) )
 * @param [Float] x
 * @return [Float] y, always between 0.0 and 1.0
 */
static VALUE sigmoid_function( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_sigmoid_function( x ) );
}


/* @overload bulk_apply_function( narray )
 * Maps an array of values. Converts arrays of single-precision floats in-place.
 * @param [NArray] narray array of input values
 * @return [NArray<sfloat>] mapped values, will be same object as narray if single-precision.
 */
static VALUE sigmoid_bulk_apply_function( VALUE self, VALUE r_narr ) {
  struct NARRAY *na_a;
  volatile VALUE val_a;

  val_a = na_cast_object(r_narr, NA_SFLOAT);
  GetNArray( val_a, na_a );

  raw_sigmoid_bulk_apply_function( na_a->total, (float*) na_a->ptr );

  return val_a;
}

/* @overload derivative( x )
 * Calculates value of dy/dx of sigmoid function, given x.
 * @param [Float] x
 * @return [Float] dy/dx, always between 0.0 and 0.5
 */
static VALUE sigmoid_derivative( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_sigmoid_derivative( x ) );
}

/* @overload derivative_at( y )
 * Calculates value of dy/dx of sigmoid function, given y.
 * @param [Float] y
 * @return [Float] dy/dx, always between 0.0 and 0.5
 */
static VALUE sigmoid_derivative_at( VALUE self, VALUE r_y ) {
  float y = NUM2FLT( r_y );
  return FLT2NUM( raw_sigmoid_derivative_at( y ) );
}

/* Document-module:  CoNeNe::Transfer::TanH
 *
 * This is a tried-and-tested transfer function which has desirable properties
 * for backpropagation training, and is used by default in CoNeNe for all hidden
 * layers. It returns y =~ -1.0 for large negative x, y = 0.0 when x= 0.0 and
 * y =~ 1.0 for large positive x.
 */

/* @overload function( x )
 * Calculates value of
 *     y =  2.0 / (1.0 + exp(-2*x) ) - 1.0
 * @param [Float] x
 * @return [Float] y, always between 0.0 and 1.0
 */
static VALUE tanh_function( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_tanh_function( x ) );
}

/* @overload bulk_apply_function( narray )
 * Maps an array of values. Converts arrays of single-precision floats in-place.
 * @param [NArray] narray array of input values
 * @return [NArray<sfloat>] mapped values, will be same object as narray if single-precision.
 */
static VALUE tanh_bulk_apply_function( VALUE self, VALUE r_narr ) {
  struct NARRAY *na_a;
  volatile VALUE val_a;

  val_a = na_cast_object(r_narr, NA_SFLOAT);
  GetNArray( val_a, na_a );

  raw_tanh_bulk_apply_function( na_a->total, (float*) na_a->ptr );

  return val_a;
}

/* @overload derivative( x )
 * Calculates value of dy/dx of tanh function, given x.
 * @param [Float] x
 * @return [Float] dy/dx, always between 0.0 and 1.0
 */
static VALUE tanh_derivative( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_tanh_derivative( x ) );
}

/* @overload derivative_at( y )
 * Calculates value of dy/dx of tanh function, given y.
 * @param [Float] y
 * @return [Float] dy/dx, always between 0.0 and 1.0
 */
static VALUE tanh_derivative_at( VALUE self, VALUE r_y ) {
  float y = NUM2FLT( r_y );
  return FLT2NUM( raw_tanh_derivative_at( y ) );
}

/* Document-module:  CoNeNe::Transfer::ReLU
 *
 * ReLU stands for "Rectifiled Linear Unit". It returns 0.0 for negative input
 * and y = x for positive values. It is fast to calculate, and for some scenarios is
 * quicker to train. It is not used by default inside CoNeNe, but
 * individual CoNeNe::MLP::Layer objects can be constructed to use it.
 */

/* @overload function( x )
 * Calculates value of
 *     y =  ( x > 0.0 ) ? x : 0.0
 * @param [Float] x
 * @return [Float] y, always 0.0 or above
 */
static VALUE relu_function( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_relu_function( x ) );
}

/* @overload bulk_apply_function( narray )
 * Maps an array of values. Converts arrays of single-precision floats in-place.
 * @param [NArray] narray array of input values
 * @return [NArray<sfloat>] mapped values, will be same object as narray if single-precision.
 */
static VALUE relu_bulk_apply_function( VALUE self, VALUE r_narr ) {
  struct NARRAY *na_a;
  volatile VALUE val_a;

  val_a = na_cast_object(r_narr, NA_SFLOAT);
  GetNArray( val_a, na_a );

  raw_relu_bulk_apply_function( na_a->total, (float*) na_a->ptr );

  return val_a;
}

/* @overload derivative( x )
 * Calculates value of dy/dx of relu function, given x.
 * @param [Float] x
 * @return [Float] dy/dx, either 0.0 or 1.0
 */
static VALUE relu_derivative( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_relu_derivative( x ) );
}

/* @overload derivative_at( y )
 * Calculates value of dy/dx of relu function, given y.
 * @param [Float] y
 * @return [Float] dy/dx, either 0.0 or 1.0
 */
static VALUE relu_derivative_at( VALUE self, VALUE r_y ) {
  float y = NUM2FLT( r_y );
  return FLT2NUM( raw_relu_derivative_at( y ) );
}


/* Document-module:  CoNeNe::Transfer::Linear
 *
 * Linear units are useful as output for regression problems. They are coded as a transfer
 * type for convenience (it is easier to have a defined type than detecting nil type and handling
 * differently)
 */

/* @overload function( x )
 * Calculates value of
 *     y =  x
 * @param [Float] x
 * @return [Float] y
 */
static VALUE linear_function( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_linear_function( x ) );
}

/* @overload bulk_apply_function( narray )
 * Maps an array of values. The implementation is short-circuited and does not proces the target
 * array.
 * @param [NArray] narray array of input values
 * @return [NArray<sfloat>] mapped values, will be same object as narray if single-precision.
 */
static VALUE linear_bulk_apply_function( VALUE self, VALUE r_narr ) {
  struct NARRAY *na_a;
  volatile VALUE val_a;

  val_a = na_cast_object(r_narr, NA_SFLOAT);
  GetNArray( val_a, na_a );

  raw_linear_bulk_apply_function( na_a->total, (float*) na_a->ptr );

  return val_a;
}

/* @overload derivative( x )
 * Calculates value of dy/dx of linear function, given x.
 * @param [Float] x
 * @return [Float] dy/dx, which should always be 1.0
 */
static VALUE linear_derivative( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_linear_derivative( x ) );
}

/* @overload derivative_at( y )
 * Calculates value of dy/dx of linear function, given y.
 * @param [Float] y
 * @return [Float] dy/dx, which should always be 1.0
 */
static VALUE linear_derivative_at( VALUE self, VALUE r_y ) {
  float y = NUM2FLT( r_y );
  return FLT2NUM( raw_linear_derivative_at( y ) );
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void init_transfer_module( ) {
  volatile VALUE conene_root = rb_define_module( "CoNeNe" );

  Transfer = rb_define_module_under( conene_root, "Transfer" );

  Sigmoid = rb_define_module_under( Transfer, "Sigmoid" );
  rb_define_singleton_method( Sigmoid, "function", sigmoid_function, 1 );
  rb_define_singleton_method( Sigmoid, "bulk_apply_function", sigmoid_bulk_apply_function, 1 );
  rb_define_singleton_method( Sigmoid, "derivative", sigmoid_derivative, 1 );
  rb_define_singleton_method( Sigmoid, "derivative_at", sigmoid_derivative_at, 1 );

  TanH = rb_define_module_under( Transfer, "TanH" );
  rb_define_singleton_method( TanH, "function", tanh_function, 1 );
  rb_define_singleton_method( TanH, "bulk_apply_function", tanh_bulk_apply_function, 1 );
  rb_define_singleton_method( TanH, "derivative", tanh_derivative, 1 );
  rb_define_singleton_method( TanH, "derivative_at", tanh_derivative_at, 1 );

  ReLU = rb_define_module_under( Transfer, "ReLU" );
  rb_define_singleton_method( ReLU, "function", relu_function, 1 );
  rb_define_singleton_method( ReLU, "bulk_apply_function", relu_bulk_apply_function, 1 );
  rb_define_singleton_method( ReLU, "derivative", relu_derivative, 1 );
  rb_define_singleton_method( ReLU, "derivative_at", relu_derivative_at, 1 );

  Linear = rb_define_module_under( Transfer, "Linear" );
  rb_define_singleton_method( Linear, "function", linear_function, 1 );
  rb_define_singleton_method( Linear, "bulk_apply_function", linear_bulk_apply_function, 1 );
  rb_define_singleton_method( Linear, "derivative", linear_derivative, 1 );
  rb_define_singleton_method( Linear, "derivative_at", linear_derivative_at, 1 );
}
