// ext/ru_ne_ne/ruby_class_learn_mbgd_layer.h

#ifndef RUBY_CLASS_LEARN_MBGD_LAYER_H
#define RUBY_CLASS_LEARN_MBGD_LAYER_H

#include <ruby.h>
#include "narray.h"
#include "struct_layer_ff.h"
#include "struct_mbgd_layer.h"
#include "shared_vars.h"
#include "ruby_c_conversions.h"

void init_mbgd_layer_class( );
void assert_value_wraps_mbgd_layer( VALUE obj );
void copy_hash_to_mbgd_layer_properties( VALUE rv_opts, MBGDLayer *mbgd_layer );

#endif
