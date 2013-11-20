// ext/con_ne_ne/struct_mlp_network.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating MLP_Network structs
//

#ifndef STRUCT_MLP_NETWORK_H
#define STRUCT_MLP_NETWORK_H

#include <ruby.h>
#include "narray.h"
#include "core_mt.h"
#include "ruby_class_mlp_layer.h"

typedef struct _mlp_network_raw {
    int num_inputs;
    int num_outputs;
    int num_layers;
    int *layer_sizes;
  } MLP_Network;

MLP_Network *p_mlp_network_create();

void p_mlp_network_destroy( MLP_Network *mlp_network );

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void p_mlp_network_gc_mark( MLP_Network *mlp_network );

#endif
