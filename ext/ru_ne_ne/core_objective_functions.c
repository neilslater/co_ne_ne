// ext/ru_ne_ne/core_objective_functions.c

#include "core_objective_functions.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of objective functions used as optimisation targets
//

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Mean Square Error
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
void obj_mse_tr_linear_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  raw_mse_delta_loss( n, predictions, targets, output_de_dz );
}

void obj_mse_tr_sigmoid_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  // This two-step process is typical of a non-optimised calculation of de_dz given by
  // chain rule:  de_dz = de_da * da_dz
  int i;
  // TODO: Initialise and re-use this for a whole training run?
  float *da_dz = xmalloc( sizeof(float) * n );

  raw_mse_delta_loss( n, predictions, targets, output_de_dz );
  raw_sigmoid_bulk_derivative_at( n, predictions, da_dz );

  for ( i = 0; i < n ; i++ ) {
    output_de_dz[i] *= da_dz[i];
  }

  xfree( da_dz );
}

void obj_mse_tr_tanh_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  int i;
  // TODO: Initialise and re-use this for a whole training run?
  float *da_dz = xmalloc( sizeof(float) * n );

  raw_mse_delta_loss( n, predictions, targets, output_de_dz );
  raw_tanh_bulk_derivative_at( n, predictions, da_dz );

  for ( i = 0; i < n ; i++ ) {
    output_de_dz[i] *= da_dz[i];
  }

  xfree( da_dz );
}

void obj_mse_tr_softmax_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  int i, k;
  float t;
  // TODO: Initialise and re-use this for a whole training run?
  float *da_dz = xmalloc( sizeof(float) * n * n );
  float *tmp_de_dz = xmalloc( sizeof(float) * n );

  raw_mse_delta_loss( n, predictions, targets, output_de_dz );

  raw_softmax_bulk_derivative_at( n, predictions, da_dz );

  for ( i = 0; i < n ; i++ ) {
    t = 0.0;
    for ( k = 0; k < n ; k++ ) {
      t += output_de_dz[k] * da_dz[i * n + k];
    }
    tmp_de_dz[i] = t;
  }

  memcpy( output_de_dz, tmp_de_dz, n * sizeof(float) );

  xfree( da_dz );
  xfree( tmp_de_dz );
}

void obj_mse_tr_relu_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  int i;
  // TODO: Initialise and re-use this for a whole training run?
  float *da_dz = xmalloc( sizeof(float) * n );

  raw_mse_delta_loss( n, predictions, targets, output_de_dz );
  raw_relu_bulk_derivative_at( n, predictions, da_dz );

  for ( i = 0; i < n ; i++ ) {
    output_de_dz[i] *= da_dz[i];
  }

  xfree( da_dz );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Log Loss / Cross-Entropy
//

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
  float p1, p2;
  int i;
  for ( i = 0; i < n ; i++ ) {
    if ( targets[i] >= 1.0 ) {
      p1 = eta > predictions[i] ? eta : predictions[i];
      p1 = p1 > 1.0 ? 1.0 : p1;
      delta_loss[i] = -1.0 / p1;
    } else if ( targets[i] <= 0.0 ){
      p2 = 1.0 - predictions[i];
      p2 = eta > p2 ? eta : p2;
      p2 = p2 > 1.0 ? 1.0 : p2;
      delta_loss[i] = 1.0 / p2;
    } else {
      p1 = eta > predictions[i] ? eta : predictions[i];
      p1 = p1 > 1.0 ? 1.0 : p1;
      p2 = 1.0 - predictions[i];
      p2 = eta > p2 ? eta : p2;
      p2 = p2 > 1.0 ? 1.0 : p2;
      delta_loss[i] = ( ( 1.0 - targets[i] ) / p2 ) - ( targets[i] / p1 );
    }
  }
  return;
}

void obj_logloss_tr_linear_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  rb_raise( rb_eRuntimeError, "Cannot combine logloss objective and linear output layer." );
}

void obj_logloss_tr_sigmoid_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  int i;

  // TODO: Initialise and re-use this for a whole training run?
  float *da_dz = xmalloc( sizeof(float) * n );

  raw_delta_logloss( n, predictions, targets, output_de_dz, 1e-15 );
  raw_sigmoid_bulk_derivative_at( n, predictions, da_dz );

  for ( i = 0; i < n ; i++ ) {
    output_de_dz[i] *= da_dz[i];
  }

  xfree( da_dz );
  return;
}

void obj_logloss_tr_tanh_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  rb_raise( rb_eRuntimeError, "Cannot combine logloss objective and tanh output layer." );
}

void obj_logloss_tr_softmax_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {

}

void obj_logloss_tr_relu_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  rb_raise( rb_eRuntimeError, "Cannot combine logloss objective and relu output layer." );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Multiclass Log Loss
//

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

// TODO: Fix this for when  targets[i] can have other values than just one item at 1.0.

void raw_delta_mlogloss( int n, float* predictions, float* targets, float* delta_loss, float eta ) {
  float p1;
  int i;
  for ( i = 0; i < n ; i++ ) {
    if ( targets[i] > 0.0 ) {
      p1 = eta > predictions[i] ? eta : predictions[i];
      p1 = p1 > 1.0 ? 1.0 : p1;
      delta_loss[i] = -targets[i] / p1;
    } else {
      delta_loss[i] = 0;
    }
  }
  return;
}
