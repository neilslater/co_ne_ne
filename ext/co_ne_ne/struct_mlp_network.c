// ext/co_ne_ne/struct_mlp_network.c

#include "struct_mlp_network.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating MLP_Network structs
//

MLP_Network *p_mlp_network_create() {
  MLP_Network *mlp_network;
  mlp_network = xmalloc( sizeof(MLP_Network) );
  mlp_network->num_layers = 0;
  mlp_network->num_outputs = 0;
  mlp_network->num_inputs = 0;

  return mlp_network;
}

void p_mlp_network_destroy( MLP_Network *mlp_network ) {
  xfree( mlp_network );
  // No need to free NArrays - they will be handled by Ruby's GC, and may still be reachable
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void p_mlp_network_gc_mark( MLP_Network *mlp_network ) {
  return;
}
