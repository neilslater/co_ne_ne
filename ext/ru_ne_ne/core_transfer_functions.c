// ext/ru_ne_ne/core_transfer_functions.c

#include "core_transfer_functions.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of transfer functions used to map activation values
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

///////////////////////////////////////////

float raw_linear_function( float x ) {
  return x;
}

void raw_linear_bulk_apply_function( int n, float *ptr ) {
  return;
}

float raw_linear_derivative( float x ) {
  return 1.0;
}

float raw_linear_derivative_at( float y ) {
  return 1.0;
}

void raw_linear_bulk_derivative_at( int n, float *func_ptr, float *deriv_ptr  ) {
  int i;
  for( i = 0; i < n; i++ ) {
    deriv_ptr[i] = 1.0;
  }
}

///////////////////////////////////////////

void raw_softmax_bulk_apply_function( int n, float *ptr ) {
  int i;
  float denom = 0.0;
  for( i = 0; i < n; i++ ) {
    ptr[i] = exp( ptr[i] );
    denom += ptr[i];
  }
  denom = 1.0/denom;
  for( i = 0; i < n; i++ ) {
    ptr[i] *= denom;
  }
  return;
}

// NB deriv_ptr needs to have size n*n as the bulk derivative
void raw_softmax_bulk_derivative_at( int n, float *func_ptr, float *deriv_ptr ) {
  int i, k;
  for( i = 0; i < n; i++ ) {
    for( k = 0; k < n; k++ ) {
      if ( i == k ) {
        deriv_ptr[i+n*k] = func_ptr[i] * ( 1.0 - func_ptr[k] );
      } else {
        deriv_ptr[i+n*k] = func_ptr[i] * ( - func_ptr[k] );
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  These intended to be called from neural-net routines in C
//

float transfer_function( transfer_type t, float x ) {
  switch ( t ) {
    case SIGMOID:
      return raw_sigmoid_function( x );
    case TANH:
      return raw_tanh_function( x );
    case RELU:
      return raw_relu_function( x );
    case LINEAR:
      return x;
    // This is not supposed to be called
    case SOFTMAX:
      return 1.0;
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
    case LINEAR:
      return;
    case SOFTMAX:
      return raw_softmax_bulk_apply_function( n, ptr );
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
    case LINEAR:
      return 1.0;
    // This is not supposed to be called
    case SOFTMAX:
      return 0.0;
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
    case LINEAR:
      return 1.0;
    // This is not supposed to be called
    case SOFTMAX:
      return 0.0;
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
    case LINEAR:
      return raw_linear_bulk_derivative_at( n, func_ptr, deriv_ptr );
    case SOFTMAX:
      return raw_softmax_bulk_derivative_at( n, func_ptr, deriv_ptr );
  }
}
