// ext/ru_ne_ne/struct_mlp_network.c

#include "struct_mlp_network.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating MLP_Network structs
//

MLP_Network *p_mlp_network_create() {
  MLP_Network *mlp_network;
  mlp_network = xmalloc( sizeof(MLP_Network) );

  mlp_network->first_layer = Qnil;
  mlp_network->eta = 1.0;
  mlp_network->momentum = 0.5;

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
  volatile VALUE layer_object;
  volatile VALUE prev_layer_object;
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
  Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
  mlp_layer->locked_input = 1;

  mlp_network->first_layer = layer_object;
  return;
}

int p_mlp_network_count_layers( MLP_Network *mlp_network ) {
  int count = 0;
  VALUE layer_object;
  MLP_Layer *mlp_layer;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    count++;
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  return count;
}

void p_mlp_network_init_layer_weights( MLP_Network *mlp_network, float min_weight, float max_weight ) {
  VALUE layer_object;
  MLP_Layer *mlp_layer;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    p_mlp_layer_init_weights( mlp_layer, min_weight, max_weight );
    layer_object = mlp_layer->output_layer;
  }

  return;
}

int p_mlp_network_num_outputs( MLP_Network *mlp_network ) {
  int count_outputs = 0;
  VALUE layer_object;
  MLP_Layer *mlp_layer;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    count_outputs = mlp_layer->num_outputs;
    layer_object = mlp_layer->output_layer;
  }

  return count_outputs;
}

int p_mlp_network_num_inputs( MLP_Network *mlp_network ) {
  MLP_Layer *mlp_layer;
  Data_Get_Struct( mlp_network->first_layer, MLP_Layer, mlp_layer );
  return mlp_layer->num_inputs;
}

// This assumes input has been assigned already to first layer
void p_mlp_network_run( MLP_Network *mlp_network ) {
  volatile VALUE layer_object;
  MLP_Layer *mlp_layer;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    p_mlp_layer_run( mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  return;
}

MLP_Layer *p_mlp_network_last_mlp_layer( MLP_Network *mlp_network ) {
  VALUE layer_object;
  MLP_Layer *mlp_layer;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  return mlp_layer;
}

void p_mlp_network_calc_output_deltas( MLP_Network *mlp_network, VALUE val_target ) {
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  struct NARRAY *na_output_slope;
  struct NARRAY *na_output_deltas;
  MLP_Layer *mlp_layer = p_mlp_network_last_mlp_layer( mlp_network );

  GetNArray( val_target, na_target );
  GetNArray( mlp_layer->narr_output, na_output );
  GetNArray( mlp_layer->narr_output_slope, na_output_slope );
  GetNArray( mlp_layer->narr_output_deltas, na_output_deltas );

  transfer_bulk_derivative_at( mlp_layer->transfer_fn, mlp_layer->num_outputs, (float *) na_output->ptr, (float *) na_output_slope->ptr );

  core_calc_output_deltas( mlp_layer->num_outputs, (float *) na_output->ptr,
      (float *) na_output_slope->ptr, (float *) na_target->ptr, (float *) na_output_deltas->ptr );

  return;
}

void p_mlp_network_backprop_deltas( MLP_Network *mlp_network ) {
  MLP_Layer *mlp_layer_input;
  MLP_Layer *mlp_layer = p_mlp_network_last_mlp_layer( mlp_network );
  while ( ! NIL_P(mlp_layer->input_layer) ) {
    Data_Get_Struct( mlp_layer->input_layer, MLP_Layer, mlp_layer_input );
    p_mlp_layer_backprop_deltas( mlp_layer, mlp_layer_input );
    mlp_layer = mlp_layer_input;
  }
  return;
}

void p_mlp_network_update_weights( MLP_Network *mlp_network ) {
  MLP_Layer *mlp_layer;
  Data_Get_Struct( mlp_network->first_layer, MLP_Layer, mlp_layer );

  while ( ! NIL_P(mlp_layer->output_layer) ) {
    p_mlp_layer_update_weights( mlp_layer, 1.0, 0.5 );
    Data_Get_Struct( mlp_layer->output_layer, MLP_Layer, mlp_layer );
  }
  p_mlp_layer_update_weights( mlp_layer, mlp_network->eta, mlp_network->momentum );
  return;
}

void p_mlp_network_train_once( MLP_Network *mlp_network, VALUE val_input, VALUE val_target ) {
  MLP_Layer *mlp_layer;

  ////////////////////////////////////////////////////////////////////////////////////
  // Attach input
  Data_Get_Struct( mlp_network->first_layer, MLP_Layer, mlp_layer );
  p_mlp_layer_set_input( mlp_layer, val_input );

  ////////////////////////////////////////////////////////////////////////////////////
  // Run the network forward
  p_mlp_network_run( mlp_network );

  ////////////////////////////////////////////////////////////////////////////////////
  // Calculate delta on output
  p_mlp_network_calc_output_deltas( mlp_network, val_target );

  ////////////////////////////////////////////////////////////////////////////////////
  // Back-propagate delta
  p_mlp_network_backprop_deltas( mlp_network );

  ////////////////////////////////////////////////////////////////////////////////////
  // Adjust weights
  p_mlp_network_update_weights( mlp_network );

  return;
}
