// ext/ru_ne_ne/core_objective_functions.c

#include "core_objective_functions.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of objective functions used as optimisation targets
//

float raw_mse_loss( int n, float* predictions, float* targets ) {
  float t = 0.0;
  int i;
  for ( i = 0; i < n ; i++ ) {
    t += (predictions[i] - targets[i]) * (predictions[i] - targets[i]);
  }
  return t * 0.5;
}

void raw_mse_delta_loss( int n, float* predictions, float* targets, float* delta_loss ) {
  int i;
  for ( i = 0; i < n ; i++ ) {
    delta_loss[i] = (predictions[i] - targets[i]);
  }
  return;
}
