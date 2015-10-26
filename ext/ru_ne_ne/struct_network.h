// ext/ru_ne_ne/struct_network.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definition for Network and declarations for its memory management
//

#ifndef STRUCT_NETWORK_H
#define STRUCT_NETWORK_H

#include <ruby.h>
#include "narray.h"
#include "struct_nn_model.h"
#include "struct_mbgd.h"

typedef struct _network_raw {
  volatile VALUE nn_model;
  volatile VALUE learn;
  } Network;

Network *network__create();

void network__destroy( Network *network );

void network__gc_mark( Network *network );

void network__init( Network *network, VALUE nn_model, VALUE learn );

void network__deep_copy( Network *network_copy, Network *network_orig );

Network * network__clone( Network *network_orig );

#endif
