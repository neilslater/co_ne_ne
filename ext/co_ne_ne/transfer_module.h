// ext/co_ne_ne/transfer_module.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of Transfer module
//

#ifndef TRANSFER_MODULE_H
#define TRANSFER_MODULE_H

#define NUM2FLT(x) ((float)NUM2DBL(x))
#define FLT2NUM(x) (rb_float_new((double)x))

typedef enum {SIGMOID, TANH, RELU} transfer_type;

void init_transfer_module( VALUE parent_module );

extern VALUE Transfer;
extern VALUE Sigmoid;
extern VALUE TanH;
extern VALUE ReLU;

float transfer_function( transfer_type t, float x );
void transfer_bulk_apply_function( transfer_type t, int n, float *ptr );
float transfer_derivative( transfer_type t, float x );
float transfer_derivative_at( transfer_type t, float y );
void transfer_bulk_derivative_at( transfer_type t, int n, float *func_ptr, float *deriv_ptr );

float raw_sigmoid_function( float x );
void raw_sigmoid_bulk_apply_function( int n, float *ptr );
float raw_sigmoid_derivative( float x );
float raw_sigmoid_derivative_at( float y );
void raw_sigmoid_bulk_derivative_at( int n, float *func_ptr, float *deriv_ptr  );

float raw_tanh_function( float x );
void raw_tanh_bulk_apply_function( int n, float *ptr );
float raw_tanh_derivative( float x );
float raw_tanh_derivative_at( float y );
void raw_tanh_bulk_derivative_at( int n, float *func_ptr, float *deriv_ptr  );

float raw_relu_function( float x );
void raw_relu_bulk_apply_function( int n, float *ptr );
float raw_relu_derivative( float x );
float raw_relu_derivative_at( float y );
void raw_relu_bulk_derivative_at( int n, float *func_ptr, float *deriv_ptr );

#endif
