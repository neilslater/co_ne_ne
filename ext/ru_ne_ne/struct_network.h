// ext/con_ne_ne/struct_network.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating Network structs
//

#ifndef STRUCT_MLP_NETWORK_H
#define STRUCT_MLP_NETWORK_H

#include <ruby.h>
#include "narray.h"
#include "mt.h"
#include "ruby_class_layer_ff.h"

typedef struct _network_raw {
    VALUE first_layer; // Arrays of layers are inferred from connectivity
    float eta;
    float momentum;
  } Network;

Network *network__create();

void network__destroy( Network *network );

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void network__gc_mark( Network *network );

void network__init_layers( Network *network, int nlayers, int *layer_sizes );

int network__count_layers( Network *network );

void network__init_layer_weights( Network *network, float min_weight, float max_weight );

int network__num_outputs( Network *network );

int network__num_inputs( Network *network );

void network__run( Network *network );

Layer_FF *network__last_layer_ff( Network *network );

void network__calc_output_deltas( Network *network, VALUE val_target );

void network__backprop_deltas( Network *network );

void network__update_weights( Network *network );

void network__train_once( Network *network, VALUE val_input, VALUE val_target );

#endif
