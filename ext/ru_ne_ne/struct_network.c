// ext/ru_ne_ne/struct_network.c

#include "struct_network.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for Network memory management
//

Network *network__create() {
  Network *network;
  network = xmalloc( sizeof(Network) );
  network->layers = NULL;
  network->num_layers = 0;
  network->num_inputs = 0;
  network->num_outputs = 0;
  return network;
}

void network__destroy( Network *network ) {
  xfree( network->layers );
  xfree( network );
  return;
}

void network__init( Network *network, int num_layers, VALUE *layers ) {
  int i, last_num_outputs;
  Layer_FF *layer_ff;

  network->num_layers = num_layers;
  network->layers = ALLOC_N( VALUE, num_layers );

  for ( i = 0; i < network->num_layers; i++ ) {
    Data_Get_Struct( layers[i], Layer_FF, layer_ff );
    if ( i == 0 ) {
      network->num_inputs = layer_ff->num_inputs;
    } else {
      if ( layer_ff->num_inputs != last_num_outputs ) {
        rb_raise( rb_eRuntimeError, "When building network, layer connections failed between output size %d and next input size %d",
            last_num_outputs, layer_ff->num_inputs );
      }
    }
    last_num_outputs = layer_ff->num_outputs;

    network->layers[i] = layers[i];
  }

  network->num_outputs = last_num_outputs;

  return;
}

void network__gc_mark( Network *network ) {
  int i;
  for ( i = 0; i < network->num_layers; i++ ) {
    rb_gc_mark( network->layers[i] );
  }
  return;
}

void network__deep_copy( Network *network_copy, Network *network_orig ) {
  network_copy->num_layers = network_orig->num_layers;
  network_copy->num_inputs = network_orig->num_inputs;
  network_copy->num_outputs = network_orig->num_outputs;

  network_copy->layers = ALLOC_N( VALUE, network_copy->num_layers );
  int i;
  for ( i = 0; i < network_copy->num_layers; i++ ) {
    // This calls .clone of each layer via Ruby
    network_copy->layers[i] = rb_funcall( network_orig->layers[i], rb_intern("clone"), 0 );
  }

  return;
}

Network * network__clone( Network *network_orig ) {
  Network * network_copy = network__create();
  network__deep_copy( network_copy, network_orig );
  return network_copy;
}
