// ext/ru_ne_ne/struct_network.c

#include "struct_network.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for Network memory management
//

Network *network__create() {
  Network *network;
  network = xmalloc( sizeof(Network) );
  network->nn_model = Qnil;
  network->learn = Qnil;
  return network;
}

void network__destroy( Network *network ) {
  xfree( network );
  return;
}

void network__gc_mark( Network *network ) {
  rb_gc_mark( network->nn_model );
  rb_gc_mark( network->learn );
  return;
}

void network__init( Network *network, VALUE nn_model, VALUE learn ) {
  network->nn_model = nn_model;
  network->learn = learn;
  return;
}

void network__deep_copy( Network *network_copy, Network *network_orig ) {
  // Deep clone current model and learner

  network_copy->nn_model = network_orig->nn_model;
  network_copy->learn = network_orig->learn;

  return;
}

Network * network__clone( Network *network_orig ) {
  Network * network_copy = network__create();
  network__deep_copy( network_copy, network_orig );
  return network_copy;
}
