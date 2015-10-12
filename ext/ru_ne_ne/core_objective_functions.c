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

// logloss objective plus sigmoid transfer optimises to (predictions - targets)
void obj_logloss_tr_sigmoid_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  int i;
  for ( i = 0; i < n ; i++ ) {
    output_de_dz[i] = predictions[i] - targets[i];
  }
  return;
}

void obj_logloss_tr_tanh_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  rb_raise( rb_eRuntimeError, "Cannot combine logloss objective and tanh output layer." );
}

void obj_logloss_tr_softmax_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  int i, k;
  float t;
  // TODO: Initialise and re-use this for a whole training run?
  float *da_dz = xmalloc( sizeof(float) * n * n );
  float *tmp_de_dz = xmalloc( sizeof(float) * n );

  raw_delta_logloss( n, predictions, targets, output_de_dz, 1e-15 );

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
    if ( targets[i] > 0.0 ) {
      p1 = eta > predictions[i] ? eta : predictions[i];
      p1 = p1 > 1.0 ? 1.0 : p1;
      t -= targets[i] * log(p1);
    }
  }
  return t;
}

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

void obj_mlogloss_tr_linear_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  rb_raise( rb_eRuntimeError, "Cannot combine mlogloss objective and linear output layer." );
}

void obj_mlogloss_tr_sigmoid_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  int i;

  // TODO: Initialise and re-use this for a whole training run?
  float *da_dz = xmalloc( sizeof(float) * n );

  raw_delta_mlogloss( n, predictions, targets, output_de_dz, 1e-15 );
  raw_sigmoid_bulk_derivative_at( n, predictions, da_dz );

  for ( i = 0; i < n ; i++ ) {
    output_de_dz[i] *= da_dz[i];
  }

  xfree( da_dz );
  return;
}

void obj_mlogloss_tr_tanh_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  rb_raise( rb_eRuntimeError, "Cannot combine mlogloss objective and tanh output layer." );
}

void obj_mlogloss_tr_softmax_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  int i,k,n_ones=0,n_zeros=0,is_simple = 1;
  float t;

  // There is an optimised case for mclass-logloss plus softmax, when there is a single target class
  // Annoyingly, detecting it takes some effort (but still worthwhile)
  for ( i = 0; i < n ; i++ ) {
    switch ( (int) targets[i] * 1000000 ) {
      case 0:
        n_zeros++;
        break;
      case 1000000:
        n_ones++;
        break;
      default:
        is_simple = 0;
    }
  }
  if ( (n_ones != 1) || (n_zeros+n_ones != n) ) {
    is_simple = 0;
  }

  // Optimised gradient is just predictions - targets
  if (is_simple) {
    for ( i = 0; i < n ; i++ ) {
      output_de_dz[i] = predictions[i] - targets[i];
    }
    return;
  }

  // Sadly this cannot be fully optimised, and we have to create more complex temporary measurements
  float *da_dz = xmalloc( sizeof(float) * n * n );
  float *tmp_de_dz = xmalloc( sizeof(float) * n );

  raw_delta_mlogloss( n, predictions, targets, output_de_dz, 1e-15 );

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

void obj_mlogloss_tr_relu_de_dz( int n, float* predictions, float* targets, float* output_de_dz ) {
  rb_raise( rb_eRuntimeError, "Cannot combine mlogloss objective and relu output layer." );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Combined de_dz function across all objectibe and transfer type combinations
//

void de_dz_from_objective_and_transfer( objective_type obj, transfer_type t, int n, float* predictions, float* targets, float* output_de_dz ) {
  switch( t ) {
    case SIGMOID:
      switch( obj ) {
        case MSE:
          obj_mse_tr_sigmoid_de_dz( n, predictions, targets, output_de_dz );
          break;
        case LOGLOSS:
          obj_logloss_tr_sigmoid_de_dz( n, predictions, targets, output_de_dz );
          break;
        case MLOGLOSS:
          obj_mlogloss_tr_sigmoid_de_dz( n, predictions, targets, output_de_dz );
          break;
      }
      break;

    case TANH:
      switch( obj ) {
        case MSE:
          obj_mse_tr_tanh_de_dz( n, predictions, targets, output_de_dz );
          break;
        case LOGLOSS:
          obj_logloss_tr_tanh_de_dz( n, predictions, targets, output_de_dz );
          break;
        case MLOGLOSS:
          obj_mlogloss_tr_tanh_de_dz( n, predictions, targets, output_de_dz );
          break;
      }
      break;

    case RELU:
      switch( obj ) {
        case MSE:
          obj_mse_tr_relu_de_dz( n, predictions, targets, output_de_dz );
          break;
        case LOGLOSS:
          obj_logloss_tr_relu_de_dz( n, predictions, targets, output_de_dz );
          break;
        case MLOGLOSS:
          obj_mlogloss_tr_relu_de_dz( n, predictions, targets, output_de_dz );
          break;
      }
      break;

    case LINEAR:
      switch( obj ) {
        case MSE:
          obj_mse_tr_linear_de_dz( n, predictions, targets, output_de_dz );
          break;
        case LOGLOSS:
          obj_logloss_tr_linear_de_dz( n, predictions, targets, output_de_dz );
          break;
        case MLOGLOSS:
          obj_mlogloss_tr_linear_de_dz( n, predictions, targets, output_de_dz );
          break;
      }
      break;

    case SOFTMAX:
      switch( obj ) {
        case MSE:
          obj_mse_tr_softmax_de_dz( n, predictions, targets, output_de_dz );
          break;
        case LOGLOSS:
          obj_logloss_tr_softmax_de_dz( n, predictions, targets, output_de_dz );
          break;
        case MLOGLOSS:
          obj_mlogloss_tr_softmax_de_dz( n, predictions, targets, output_de_dz );
          break;
      }
      break;
  }

  return;
}

float objective_function_loss( objective_type obj, int n, float* predictions, float* targets ) {
  switch( obj ) {
  case MSE:
    return raw_mse_loss( n, predictions, targets );
  case LOGLOSS:
    return raw_logloss( n, predictions, targets, 1e-15 );
  case MLOGLOSS:
    return raw_mlogloss( n, predictions, targets, 1e-15 );
  }
  return 0.0;
}
