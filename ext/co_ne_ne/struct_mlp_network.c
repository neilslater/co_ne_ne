// ext/co_ne_ne/struct_mlp_network.c

#include "struct_mlp_network.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating MLP_Network structs
//

MLP_Network *p_mlp_network_create() {
  MLP_Network *mlp_network;
  mlp_network = xmalloc( sizeof(MLP_Network) );

  mlp_network->first_layer = Qnil;

  return mlp_network;
}

void p_mlp_network_destroy( MLP_Network *mlp_network ) {
  xfree( mlp_network );
  // VALUEs are not cleared here, they need to be left for GC to handle
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void p_mlp_network_gc_mark( MLP_Network *mlp_network ) {
  rb_gc_mark( mlp_network->first_layer );
  return;
}

void p_mlp_network_init_layers( MLP_Network *mlp_network, int nlayers, int *layer_sizes ) {
  MLP_Layer *mlp_layer, *mlp_layer_prev;
  VALUE layer_object;
  VALUE prev_layer_object;
  int i;

  // build layers backwards
  layer_object = mlp_layer_new_ruby_object( layer_sizes[nlayers-1], layer_sizes[nlayers], SIGMOID );
  for( i = nlayers - 1; i > 0; i-- ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    prev_layer_object =  mlp_layer_new_ruby_object( layer_sizes[i-1], layer_sizes[i], TANH );
    Data_Get_Struct( prev_layer_object, MLP_Layer, mlp_layer_prev );

    mlp_layer->narr_input = mlp_layer_prev->narr_output;
    mlp_layer->input_layer = prev_layer_object;
    mlp_layer_prev->output_layer = layer_object;
    layer_object = prev_layer_object;
  }

  mlp_network->first_layer = layer_object;
  return;
}