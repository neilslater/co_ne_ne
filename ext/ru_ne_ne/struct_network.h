// ext/con_ne_ne/struct_network.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating MLP_Network structs
//

#ifndef STRUCT_MLP_NETWORK_H
#define STRUCT_MLP_NETWORK_H

#include <ruby.h>
#include "narray.h"
#include "core_mt.h"
#include "ruby_class_layer_ff.h"

typedef struct _network_raw {
    VALUE first_layer; // Arrays of layers are inferred from connectivity
    float eta;
    float momentum;
  } MLP_Network;

MLP_Network *p_network_create();

void p_network_destroy( MLP_Network *network );

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void p_network_gc_mark( MLP_Network *network );

void p_network_init_layers( MLP_Network *network, int nlayers, int *layer_sizes );

int p_network_count_layers( MLP_Network *network );

void p_network_init_layer_weights( MLP_Network *network, float min_weight, float max_weight );

int p_network_num_outputs( MLP_Network *network );

int p_network_num_inputs( MLP_Network *network );

void p_network_run( MLP_Network *network );

s_Layer_FF *p_network_last_layer_ff( MLP_Network *network );

void p_network_calc_output_deltas( MLP_Network *network, VALUE val_target );

void p_network_backprop_deltas( MLP_Network *network );

void p_network_update_weights( MLP_Network *network );

void p_network_train_once( MLP_Network *network, VALUE val_input, VALUE val_target );

#endif
