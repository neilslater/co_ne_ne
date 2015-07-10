// ext/ru_ne_ne/ruby_module_transfer.c

#include "ruby_module_transfer.h"

/* Document-module:  RuNeNe::Transfer::Sigmoid
 *
 * This is a tried-and-tested transfer function which has desirable properties
 * for backpropagation training. It is used by default in RuNeNe for the last (output)
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

/* Document-module:  RuNeNe::Transfer::TanH
 *
 * This is a tried-and-tested transfer function which has desirable properties
 * for backpropagation training, and is used by default in RuNeNe for all hidden
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

/* Document-module:  RuNeNe::Transfer::ReLU
 *
 * ReLU stands for "Rectifiled Linear Unit". It returns 0.0 for negative input
 * and y = x for positive values. It is fast to calculate, and for some scenarios is
 * quicker to train. It is not used by default inside RuNeNe, but
 * individual RuNeNe::Layer::FeedForward objects can be constructed to use it.
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


/* Document-module:  RuNeNe::Transfer::Linear
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

/* Document-module:  RuNeNe::Transfer::Softmax
 *
 * Softmax is used for multi-class classification where an example can only be in one of
 * many possible classes. It is usually paired with the MulticlassLogLoss objective function.
 *
 * Softmax is applied to a whole vector at once, so only bulk methods are available.
 */


/* @overload bulk_apply_function( narray )
 * Maps an array of values.
 * @param [NArray] narray array of input values
 * @return [NArray<sfloat>] mapped values.
 */
static VALUE softmax_bulk_apply_function( VALUE self, VALUE r_narr ) {
  struct NARRAY *na_a;
  volatile VALUE val_a;

  val_a = na_cast_object(r_narr, NA_SFLOAT);
  GetNArray( val_a, na_a );

  raw_softmax_bulk_apply_function( na_a->total, (float*) na_a->ptr );

  return val_a;
}

/* @overload bulk_derivative_at( narray )
 * Maps an array of values.
 * @param [NArray] narray array of input values
 * @return [NArray<sfloat>] mapped values.
 */

static VALUE softmax_bulk_derivative_at( VALUE self, VALUE r_narr ) {
  struct NARRAY *na_a;
  volatile VALUE val_a;

  struct NARRAY *na_b;
  volatile VALUE val_b;
  int softmax_deriv_shape[2];

  val_a = na_cast_object(r_narr, NA_SFLOAT);
  GetNArray( val_a, na_a );

  softmax_deriv_shape[0] = na_a->total;
  softmax_deriv_shape[1] = na_a->total;

  val_b = na_make_object( NA_SFLOAT, 2, softmax_deriv_shape, cNArray );
  GetNArray( val_b, na_b );

  raw_softmax_bulk_derivative_at( na_a->total, (float*) na_a->ptr, (float*) na_b->ptr );

  return val_b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_transfer_module( ) {
  rb_define_singleton_method( RuNeNe_Transfer_Sigmoid, "function", sigmoid_function, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_Sigmoid, "bulk_apply_function", sigmoid_bulk_apply_function, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_Sigmoid, "derivative", sigmoid_derivative, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_Sigmoid, "derivative_at", sigmoid_derivative_at, 1 );

  rb_define_singleton_method( RuNeNe_Transfer_TanH, "function", tanh_function, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_TanH, "bulk_apply_function", tanh_bulk_apply_function, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_TanH, "derivative", tanh_derivative, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_TanH, "derivative_at", tanh_derivative_at, 1 );

  rb_define_singleton_method( RuNeNe_Transfer_ReLU, "function", relu_function, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_ReLU, "bulk_apply_function", relu_bulk_apply_function, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_ReLU, "derivative", relu_derivative, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_ReLU, "derivative_at", relu_derivative_at, 1 );

  rb_define_singleton_method( RuNeNe_Transfer_Linear, "function", linear_function, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_Linear, "bulk_apply_function", linear_bulk_apply_function, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_Linear, "derivative", linear_derivative, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_Linear, "derivative_at", linear_derivative_at, 1 );

  rb_define_singleton_method( RuNeNe_Transfer_Softmax, "bulk_apply_function", softmax_bulk_apply_function, 1 );
  rb_define_singleton_method( RuNeNe_Transfer_Softmax, "bulk_derivative_at", softmax_bulk_derivative_at, 1 );
}
