// ext/ru_ne_ne/struct_network.c

#include "struct_network.h"

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
  s_Layer_FF *layer_ff, *layer_ff_prev;
  volatile VALUE layer_object;
  volatile VALUE prev_layer_object;
  int i;

  // build layers backwards
  layer_object = layer_ff_new_ruby_object( layer_sizes[nlayers-1], layer_sizes[nlayers], SIGMOID );
  for( i = nlayers - 1; i > 0; i-- ) {
    Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );
    prev_layer_object =  layer_ff_new_ruby_object( layer_sizes[i-1], layer_sizes[i], TANH );
    Data_Get_Struct( prev_layer_object, s_Layer_FF, layer_ff_prev );

    layer_ff->narr_input = layer_ff_prev->narr_output;
    layer_ff->input_layer = prev_layer_object;
    layer_ff_prev->output_layer = layer_object;
    layer_object = prev_layer_object;
  }
  Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );
  layer_ff->locked_input = 1;

  mlp_network->first_layer = layer_object;
  return;
}

int p_mlp_network_count_layers( MLP_Network *mlp_network ) {
  int count = 0;
  VALUE layer_object;
  s_Layer_FF *layer_ff;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    count++;
    Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );
    layer_object = layer_ff->output_layer;
  }

  return count;
}

void p_mlp_network_init_layer_weights( MLP_Network *mlp_network, float min_weight, float max_weight ) {
  VALUE layer_object;
  s_Layer_FF *layer_ff;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );
    p_layer_ff_init_weights( layer_ff, min_weight, max_weight );
    layer_object = layer_ff->output_layer;
  }

  return;
}

int p_mlp_network_num_outputs( MLP_Network *mlp_network ) {
  int count_outputs = 0;
  VALUE layer_object;
  s_Layer_FF *layer_ff;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );
    count_outputs = layer_ff->num_outputs;
    layer_object = layer_ff->output_layer;
  }

  return count_outputs;
}

int p_mlp_network_num_inputs( MLP_Network *mlp_network ) {
  s_Layer_FF *layer_ff;
  Data_Get_Struct( mlp_network->first_layer, s_Layer_FF, layer_ff );
  return layer_ff->num_inputs;
}

// This assumes input has been assigned already to first layer
void p_mlp_network_run( MLP_Network *mlp_network ) {
  volatile VALUE layer_object;
  s_Layer_FF *layer_ff;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );
    p_layer_ff_run( layer_ff );
    layer_object = layer_ff->output_layer;
  }

  return;
}

s_Layer_FF *p_mlp_network_last_layer_ff( MLP_Network *mlp_network ) {
  VALUE layer_object;
  s_Layer_FF *layer_ff;

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );
    layer_object = layer_ff->output_layer;
  }

  return layer_ff;
}

void p_mlp_network_calc_output_deltas( MLP_Network *mlp_network, VALUE val_target ) {
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  struct NARRAY *na_output_slope;
  struct NARRAY *na_output_deltas;
  s_Layer_FF *layer_ff = p_mlp_network_last_layer_ff( mlp_network );

  GetNArray( val_target, na_target );
  GetNArray( layer_ff->narr_output, na_output );
  GetNArray( layer_ff->narr_output_slope, na_output_slope );
  GetNArray( layer_ff->narr_output_deltas, na_output_deltas );

  transfer_bulk_derivative_at( layer_ff->transfer_fn, layer_ff->num_outputs, (float *) na_output->ptr, (float *) na_output_slope->ptr );

  core_calc_output_deltas( layer_ff->num_outputs, (float *) na_output->ptr,
      (float *) na_output_slope->ptr, (float *) na_target->ptr, (float *) na_output_deltas->ptr );

  return;
}

void p_mlp_network_backprop_deltas( MLP_Network *mlp_network ) {
  s_Layer_FF *layer_ff_input;
  s_Layer_FF *layer_ff = p_mlp_network_last_layer_ff( mlp_network );
  while ( ! NIL_P(layer_ff->input_layer) ) {
    Data_Get_Struct( layer_ff->input_layer, s_Layer_FF, layer_ff_input );
    p_layer_ff_backprop_deltas( layer_ff, layer_ff_input );
    layer_ff = layer_ff_input;
  }
  return;
}

void p_mlp_network_update_weights( MLP_Network *mlp_network ) {
  s_Layer_FF *layer_ff;
  Data_Get_Struct( mlp_network->first_layer, s_Layer_FF, layer_ff );

  while ( ! NIL_P(layer_ff->output_layer) ) {
    p_layer_ff_update_weights( layer_ff, 1.0, 0.5 );
    Data_Get_Struct( layer_ff->output_layer, s_Layer_FF, layer_ff );
  }
  p_layer_ff_update_weights( layer_ff, mlp_network->eta, mlp_network->momentum );
  return;
}

void p_mlp_network_train_once( MLP_Network *mlp_network, VALUE val_input, VALUE val_target ) {
  s_Layer_FF *layer_ff;

  ////////////////////////////////////////////////////////////////////////////////////
  // Attach input
  Data_Get_Struct( mlp_network->first_layer, s_Layer_FF, layer_ff );
  p_layer_ff_set_input( layer_ff, val_input );

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
