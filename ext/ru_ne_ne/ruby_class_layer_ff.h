// ext/ru_ne_ne/ruby_class_layer_ff.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of MLP classes
//

#ifndef RUBY_CLASS_MLP_LAYER_H
#define RUBY_CLASS_MLP_LAYER_H

#include <ruby.h>
#include "narray.h"
#include "struct_layer_ff.h"
#include "ruby_module_transfer.h"
#include "shared_vars.h"

void init_layer_ff_class();
VALUE layer_ff_new_ruby_object( int n_inputs, int n_outputs, transfer_type tfn );
VALUE layer_ff_new_ruby_object_from_weights( VALUE weights, transfer_type tfn );
VALUE layer_ff_clone_ruby_object( VALUE orig );
void assert_value_wraps_layer_ff( VALUE obj );

#endif
