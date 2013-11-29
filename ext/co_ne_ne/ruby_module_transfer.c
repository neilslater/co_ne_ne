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

/*
 * Document-module:  CoNeNe::Transfer::Sigmoid
 *
 * This is a tried-and-tested transfer function which has desirable properties
 * for backpropagation training.
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

static VALUE sigmoid_derivative( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_sigmoid_derivative( x ) );
}

static VALUE sigmoid_derivative_at( VALUE self, VALUE r_y ) {
  float y = NUM2FLT( r_y );
  return FLT2NUM( raw_sigmoid_derivative_at( y ) );
}


static VALUE tanh_function( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_tanh_function( x ) );
}

static VALUE tanh_bulk_apply_function( VALUE self, VALUE r_narr ) {
  struct NARRAY *na_a;
  volatile VALUE val_a;

  val_a = na_cast_object(r_narr, NA_SFLOAT);
  GetNArray( val_a, na_a );

  raw_tanh_bulk_apply_function( na_a->total, (float*) na_a->ptr );

  return val_a;
}

static VALUE tanh_derivative( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_tanh_derivative( x ) );
}

static VALUE tanh_derivative_at( VALUE self, VALUE r_y ) {
  float y = NUM2FLT( r_y );
  return FLT2NUM( raw_tanh_derivative_at( y ) );
}


static VALUE relu_function( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_relu_function( x ) );
}

static VALUE relu_bulk_apply_function( VALUE self, VALUE r_narr ) {
  struct NARRAY *na_a;
  volatile VALUE val_a;

  val_a = na_cast_object(r_narr, NA_SFLOAT);
  GetNArray( val_a, na_a );

  raw_relu_bulk_apply_function( na_a->total, (float*) na_a->ptr );

  return val_a;
}

static VALUE relu_derivative( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_relu_derivative( x ) );
}

static VALUE relu_derivative_at( VALUE self, VALUE r_y ) {
  float y = NUM2FLT( r_y );
  return FLT2NUM( raw_relu_derivative_at( y ) );
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
}
