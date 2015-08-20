// ext/ru_ne_ne/ruby_c_conversions.c

#include "ruby_c_conversions.h"

transfer_type symbol_to_transfer_type( VALUE rv_transfer_type ) {
  ID tfn_id;

  tfn_id = rb_intern("sigmoid");
  if ( ! NIL_P(rv_transfer_type) ) {
    if ( TYPE(rv_transfer_type) != T_SYMBOL ) {
      rb_raise( rb_eTypeError, "Expected symbol for transfer function type" );
    }
    tfn_id = SYM2ID(rv_transfer_type);
  }

  if ( rb_intern("sigmoid") == tfn_id ) {
    return SIGMOID;
  } else if ( rb_intern("tanh") == tfn_id ) {
    return TANH;
  } else if ( rb_intern("relu") == tfn_id ) {
    return RELU;
  } else if ( rb_intern("linear") == tfn_id ) {
    return LINEAR;
  } else if ( rb_intern("softmax") == tfn_id ) {
    return SOFTMAX;
  } else {
    rb_raise( rb_eArgError, "Transfer function type %s not recognised", rb_id2name(tfn_id) );
  }
}

VALUE transfer_type_to_module( transfer_type t ) {
  switch ( t ) {
    case SIGMOID:
      return RuNeNe_Transfer_Sigmoid;
    case TANH:
      return RuNeNe_Transfer_TanH;
    case RELU:
      return RuNeNe_Transfer_ReLU;
    case LINEAR:
      return RuNeNe_Transfer_Linear;
    case SOFTMAX:
      return RuNeNe_Transfer_Softmax;
  }
}

VALUE transfer_type_to_symbol( transfer_type t ) {
  switch ( t ) {
    case SIGMOID:
      return ID2SYM( rb_intern("sigmoid") );
    case TANH:
      return ID2SYM( rb_intern("tanh") );
    case RELU:
      return ID2SYM( rb_intern("relu") );
    case LINEAR:
      return ID2SYM( rb_intern("linear") );
    case SOFTMAX:
      return ID2SYM( rb_intern("softmax") );
  }
}


objective_type symbol_to_objective_type( VALUE rv_objective_type ) {
  if ( TYPE(rv_objective_type) != T_SYMBOL ) {
    rb_raise( rb_eTypeError, "Expected symbol for objective function type" );
  }
  ID ofn_id = SYM2ID(rv_objective_type);
  if ( rb_intern("mse") == ofn_id ) {
    return MSE;
  } else if ( rb_intern("logloss") == ofn_id ) {
    return LOGLOSS;
  } else if ( rb_intern("mlogloss") == ofn_id ) {
    return MLOGLOSS;
  } else {
    rb_raise( rb_eArgError, "Objective function type %s not recognised", rb_id2name(ofn_id) );
  }
}

VALUE objective_type_to_module( objective_type o ) {
  switch ( o ) {
    case MSE:
      return RuNeNe_Objective_MeanSquaredError;
    case LOGLOSS:
      return RuNeNe_Objective_LogLoss;
    case MLOGLOSS:
      return RuNeNe_Objective_MulticlassLogLoss;
  }
}

VALUE objective_type_to_symbol( objective_type o ) {
  switch ( o ) {
    case MSE:
      return ID2SYM( rb_intern("mse") );
    case LOGLOSS:
      return ID2SYM( rb_intern("logloss") );
    case MLOGLOSS:
      return ID2SYM( rb_intern("mlogloss") );
  }
}

gd_accel_type symbol_to_gd_accel_type( VALUE rv_gdaccel_symbol ) {
  ID accel_id;

  accel_id = rb_intern("none");
  if ( ! NIL_P(rv_gdaccel_symbol) ) {
    if ( TYPE(rv_gdaccel_symbol) != T_SYMBOL ) {
      rb_raise( rb_eTypeError, "Gradient descent acceleration type must be a Symbol" );
    }
    accel_id = SYM2ID(rv_gdaccel_symbol);
  }

  if ( rb_intern("none") == accel_id ) {
    return GDACCEL_TYPE_NONE;
  } else if ( rb_intern("momentum") == accel_id ) {
    return GDACCEL_TYPE_MOMENTUM;
  } else if ( rb_intern("rmsprop") == accel_id ) {
    return GDACCEL_TYPE_RMSPROP;
  } else {
    rb_raise( rb_eArgError, "gd_accel_type %s not recognised", rb_id2name(accel_id) );
  }
}

VALUE gd_accel_type_to_symbol( gd_accel_type g ) {
  switch( g ) {
    case GDACCEL_TYPE_NONE:
      return ID2SYM( rb_intern("none") );
    case GDACCEL_TYPE_MOMENTUM:
      return ID2SYM( rb_intern("momentum") );
    case GDACCEL_TYPE_RMSPROP:
      return ID2SYM( rb_intern("rmsprop") );
    default:
      rb_raise( rb_eRuntimeError, "gd_accel_type not valid, internal error");
  }
}