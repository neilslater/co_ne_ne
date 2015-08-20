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
