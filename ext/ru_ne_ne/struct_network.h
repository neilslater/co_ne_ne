// ext/ru_ne_ne/struct_network.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definition for Network and declarations for its memory management
//

#ifndef STRUCT_NETWORK_H
#define STRUCT_NETWORK_H

#include <ruby.h>
#include "narray.h"
#include "struct_layer_ff.h"

typedef struct _network_raw {
  VALUE *layers;
  float **activations;
  int num_layers;
  int num_inputs;
  int num_outputs;
  } Network;

Network *network__create();

void network__init( Network *network, int num_layers, VALUE *layers );

void network__destroy( Network *network );

void network__gc_mark( Network *network );

void network__deep_copy( Network *network_copy, Network *network_orig );

Network * network__clone( Network *network_orig );

#endif
