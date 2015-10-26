// ext/ru_ne_ne/ruby_class_nn_model.h

#ifndef RUBY_CLASS_NN_MODEL_H
#define RUBY_CLASS_NN_MODEL_H

#include <ruby.h>
#include "narray.h"
#include "struct_nn_model.h"
#include "shared_vars.h"
#include "ruby_class_layer_ff.h"

void init_nn_model_class( );
NNModel *safe_get_nn_model_struct( VALUE obj );
void assert_value_wraps_nn_model( VALUE obj );

#endif
