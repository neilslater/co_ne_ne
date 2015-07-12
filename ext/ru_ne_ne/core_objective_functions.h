// ext/ru_ne_ne/core_objective_functions.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of Objective module
//

#ifndef CORE_OBJECTIVE_FUNCTIONS_H
#define CORE_OBJECTIVE_FUNCTIONS_H

#include <math.h>
#include <ruby.h>
#include "core_transfer_functions.h"

typedef enum {MSE,LOGLOSS,MLOGLOSS} objective_type;

float raw_mse_loss( int n, float* predictions, float* targets );
void raw_mse_delta_loss( int n, float* predictions, float* targets, float* delta_loss );

void obj_mse_tr_linear_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_mse_tr_sigmoid_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_mse_tr_tanh_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_mse_tr_softmax_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_mse_tr_relu_de_dz( int n, float* predictions, float* targets, float* output_de_dz );

float raw_logloss( int n, float* predictions, float* targets, float eta );
void raw_delta_logloss( int n, float* predictions, float* targets, float* delta_loss, float eta );

void obj_logloss_tr_linear_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_logloss_tr_sigmoid_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_logloss_tr_tanh_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_logloss_tr_softmax_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_logloss_tr_relu_de_dz( int n, float* predictions, float* targets, float* output_de_dz );

float raw_mlogloss( int n, float* predictions, float* targets, float eta );
void raw_delta_mlogloss( int n, float* predictions, float* targets, float* delta_loss, float eta );

void obj_mlogloss_tr_linear_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_mlogloss_tr_sigmoid_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_mlogloss_tr_tanh_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_mlogloss_tr_softmax_de_dz( int n, float* predictions, float* targets, float* output_de_dz );
void obj_mlogloss_tr_relu_de_dz( int n, float* predictions, float* targets, float* output_de_dz );

#endif
