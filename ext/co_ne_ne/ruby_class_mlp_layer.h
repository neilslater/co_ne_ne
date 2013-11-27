// ext/co_ne_ne/ruby_class_mlp_layer.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of MLP classes
//

#ifndef RUBY_CLASS_MLP_LAYER_H
#define RUBY_CLASS_MLP_LAYER_H

#include <ruby.h>
#include "narray.h"
#include "struct_mlp_layer.h"
#include "ruby_module_transfer.h"

void init_mlp_layer_class( VALUE parent );
VALUE mlp_layer_new_ruby_object( int n_inputs, int n_outputs, transfer_type tfn );
VALUE mlp_layer_new_ruby_object_from_weights( VALUE weights, transfer_type tfn );
VALUE mlp_layer_clone_ruby_object( VALUE orig );
void assert_value_wraps_mlp_layer( VALUE obj );

extern VALUE MLP;
extern VALUE Layer;
extern VALUE Network;

#endif
