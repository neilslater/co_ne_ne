// ext/co_ne_ne/transfer_module.c

#include <ruby.h>
#include "narray.h"
#include <stdio.h>
#include <xmmintrin.h>

#include "narray_shared.h"
#include "transfer_module.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// C implementations (not exported, yet!)
//

float raw_sigmoid_function( float x ) {
  return 1.0 / ( 1.0 + exp( -x ) );
}

void raw_sigmoid_bulk_apply_function( int n, float *ptr ) {
  int i;
  for( i = 0; i < n; i++ ) {
    ptr[i] = 1.0 / ( 1.0 + exp( -ptr[i] ) );
  }
}

float raw_sigmoid_derivative( float x ) {
  float y = 1.0 / ( 1.0 + exp( -x ) );
  return y * ( 1.0 - y );
}

float raw_sigmoid_derivative_at( float y ) {
  return y * ( 1.0 - y );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// To hold the module objects
//

VALUE Transfer = Qnil;
VALUE Sigmoid = Qnil;

static VALUE sigmoid_function( VALUE self, VALUE r_x ) {
  float x = NUM2FLT( r_x );
  return FLT2NUM( raw_sigmoid_function( x ) );
}

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

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_transfer_module( VALUE parent_module ) {
  Transfer = rb_define_module_under( parent_module, "Transfer" );
  Sigmoid = rb_define_module_under( Transfer, "Sigmoid" );
  rb_define_singleton_method( Sigmoid, "function", sigmoid_function, 1 );
  rb_define_singleton_method( Sigmoid, "bulk_apply_function", sigmoid_bulk_apply_function, 1 );
  rb_define_singleton_method( Sigmoid, "derivative", sigmoid_derivative, 1 );
  rb_define_singleton_method( Sigmoid, "derivative_at", sigmoid_derivative_at, 1 );
}
