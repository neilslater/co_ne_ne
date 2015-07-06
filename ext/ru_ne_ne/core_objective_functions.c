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

// Linear sub-type just for completeness
void raw_mse_de_dz_linear( int n, float* predictions, float* targets, float* delta_loss ) {
  raw_mse_delta_loss( n, predictions, targets, delta_loss );
}


float raw_logloss( int n, float* predictions, float* targets, float eta ) {
  float p1, p2, t = 0.0;
  int i;
  for ( i = 0; i < n ; i++ ) {
    p1 = eta > predictions[i] ? eta : predictions[i];
    p1 = p1 > 1.0 ? 1.0 : p1;
    p2 = 1.0 - predictions[i];
    p2 = eta > p2 ? eta : p2;
    p2 = p2 > 1.0 ? 1.0 : p2;
    t -= targets[i] * log(p1) + (1.0 - targets[i]) * log(p2);
  }
  return t;
}

void raw_delta_logloss( int n, float* predictions, float* targets, float* delta_loss, float eta ) {
  float p1;
  int i;
  for ( i = 0; i < n ; i++ ) {
    if ( targets[i] >= 1.0 ) {
      p1 = eta > predictions[i] ? eta : predictions[i];
      p1 = p1 > 1.0 ? 1.0 : p1;
      delta_loss[i] = -1.0 / p1;
    } else {
      p1 = 1.0 - predictions[i];
      p1 = eta > p1 ? eta : p1;
      p1 = p1 > 1.0 ? 1.0 : p1;
      delta_loss[i] = 1.0 / p1;
    }
  }
  return;
}

float raw_mlogloss( int n, float* predictions, float* targets, float eta ) {
  float p1, t = 0.0;
  int i;
  for ( i = 0; i < n ; i++ ) {
    p1 = eta > predictions[i] ? eta : predictions[i];
    p1 = p1 > 1.0 ? 1.0 : p1;
    t -= targets[i] * log(p1);
  }
  return t;
}

void raw_delta_mlogloss( int n, float* predictions, float* targets, float* delta_loss, float eta ) {
  float p1;
  int i;
  for ( i = 0; i < n ; i++ ) {
    if ( targets[i] >= 1.0 ) {
      p1 = eta > predictions[i] ? eta : predictions[i];
      p1 = p1 > 1.0 ? 1.0 : p1;
      delta_loss[i] = -1.0 / p1;
    } else {
      delta_loss[i] = 0;
    }
  }
  return;
}
