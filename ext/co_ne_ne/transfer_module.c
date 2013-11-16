// ext/co_ne_ne/transfer_module.c

#include <ruby.h>
#include "narray.h"
#include <stdio.h>
#include <xmmintrin.h>

#include "narray_shared.h"
#include "transfer_module.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// C implementations
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

void raw_sigmoid_bulk_derivative_at( int n, float *func_ptr, float *deriv_ptr ) {
  int i;
  for( i = 0; i < n; i++ ) {
    deriv_ptr[i] = func_ptr[i] * ( 1.0 - func_ptr[i] );
  }
}

///////////////////////////////////////////

float raw_tanh_function( float x ) {
  return ( 2.0 / (1.0 + exp(-2*x) ) ) - 1.0;
}

void raw_tanh_bulk_apply_function( int n, float *ptr ) {
  int i;
  for( i = 0; i < n; i++ ) {
    ptr[i] = ( 2.0 / (1.0 + exp(-2*ptr[i]) ) ) - 1.0;
  }
}

float raw_tanh_derivative( float x ) {
  float y = ( 2.0 / (1.0 + exp(-2*x) ) ) - 1.0;
  return 1.0 - y * y;
}

float raw_tanh_derivative_at( float y ) {
  return 1.0 - y * y;
}

void raw_tanh_bulk_derivative_at( int n, float *func_ptr, float *deriv_ptr ) {
  int i;
  for( i = 0; i < n; i++ ) {
    deriv_ptr [i] =  1.0 - func_ptr[i] * func_ptr[i];
  }
}

///////////////////////////////////////////

float raw_relu_function( float x ) {
  return x > 0.0 ? x : 0.0;
}

void raw_relu_bulk_apply_function( int n, float *ptr ) {
  int i;
  for( i = 0; i < n; i++ ) {
    ptr[i] = ( ptr[i] > 0.0 ? ptr[i] : 0.0 );
  }
}

float raw_relu_derivative( float x ) {
  return x > 0.0 ? 1.0 : 0.0;
}

float raw_relu_derivative_at( float y ) {
  return y > 0.0 ? 1.0 : 0.0;
}

void raw_relu_bulk_derivative_at( int n, float *func_ptr, float *deriv_ptr  ) {
  int i;
  for( i = 0; i < n; i++ ) {
    deriv_ptr[i] = ( func_ptr[i] > 0.0 ? 1.0 : 0.0 );
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// These intended to be called from C and switch between sub-types
//

float transfer_function( transfer_type t, float x ) {
  switch ( t ) {
    case SIGMOID:
      return raw_sigmoid_function( x );
    case TANH:
      return raw_tanh_function( x );
    case RELU:
      return raw_relu_function( x );
  }
}

void transfer_bulk_apply_function( transfer_type t, int n, float *ptr ) {
  switch ( t ) {
    case SIGMOID:
      return raw_sigmoid_bulk_apply_function( n, ptr );
    case TANH:
      return raw_tanh_bulk_apply_function( n, ptr );
    case RELU:
      return raw_relu_bulk_apply_function( n, ptr );
  }
}

float transfer_derivative( transfer_type t, float x ) {
  switch ( t ) {
    case SIGMOID:
      return raw_sigmoid_derivative( x );
    case TANH:
      return raw_tanh_derivative( x );
    case RELU:
      return raw_relu_derivative( x );
  }
}

float transfer_derivative_at( transfer_type t, float y ) {
  switch ( t ) {
    case SIGMOID:
      return raw_sigmoid_derivative_at( y );
    case TANH:
      return raw_tanh_derivative_at( y );
    case RELU:
      return raw_relu_derivative_at( y );
  }
}

void transfer_bulk_derivative_at( transfer_type t, int n, float *func_ptr, float *deriv_ptr  ) {
  switch ( t ) {
    case SIGMOID:
      return raw_sigmoid_bulk_derivative_at( n, func_ptr, deriv_ptr );
    case TANH:
      return raw_tanh_bulk_derivative_at( n, func_ptr, deriv_ptr );
    case RELU:
      return raw_relu_bulk_derivative_at( n, func_ptr, deriv_ptr );
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//
// To hold the module objects
//

VALUE Transfer = Qnil;
VALUE Sigmoid = Qnil;
VALUE TanH = Qnil;
VALUE ReLU = Qnil;

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

void init_transfer_module( VALUE parent_module ) {
  Transfer = rb_define_module_under( parent_module, "Transfer" );

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
