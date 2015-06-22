// ext/ru_ne_ne/core_objective_functions.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of Objective module
//

#ifndef CORE_OBJECTIVE_FUNCTIONS_H
#define CORE_OBJECTIVE_FUNCTIONS_H

#include <math.h>

typedef enum {MSE} objective_type;

float raw_mse_loss( int n, float* predictions, float* targets );
void raw_mse_delta_loss( int n, float* predictions, float* targets, float* delta_loss );

#endif
