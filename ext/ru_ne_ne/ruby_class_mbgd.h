// ext/ru_ne_ne/ruby_class_mbgd.h

#ifndef RUBY_CLASS_MBGD_H
#define RUBY_CLASS_MBGD_H

#include <ruby.h>
#include "narray.h"
#include "struct_mbgd.h"
#include "shared_vars.h"
#include "ruby_class_learn_mbgd_layer.h"
#include "ruby_class_nn_model.h"
#include "ruby_class_dataset.h"

void init_mbgd_class( );
void assert_value_wraps_mbgd( VALUE obj );

#endif
