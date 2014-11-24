// ext/ru_ne_ne/struct_network.c

#include "struct_network.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating Network structs
//

Network *network__create() {
  Network *network;
  network = xmalloc( sizeof(Network) );

  network->first_layer = Qnil;
  network->eta = 1.0;
  network->momentum = 0.5;

  return network;
}

void network__destroy( Network *network ) {
  xfree( network );
  // VALUEs are not cleared here, they need to be left for GC to handle
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void network__gc_mark( Network *network ) {
  rb_gc_mark( network->first_layer );
  return;
}

void network__init_layers( Network *network, int nlayers, int *layer_sizes ) {
  Layer_FF *layer_ff, *layer_ff_prev;
  volatile VALUE layer_object;
  volatile VALUE prev_layer_object;
  int i;

  // build layers backwards
  layer_object = layer_ff_new_ruby_object( layer_sizes[nlayers-1], layer_sizes[nlayers], SIGMOID );
  for( i = nlayers - 1; i > 0; i-- ) {
    Data_Get_Struct( layer_object, Layer_FF, layer_ff );
    prev_layer_object =  layer_ff_new_ruby_object( layer_sizes[i-1], layer_sizes[i], TANH );
    Data_Get_Struct( prev_layer_object, Layer_FF, layer_ff_prev );

    layer_ff->narr_input = layer_ff_prev->narr_output;
    layer_ff->input_layer = prev_layer_object;
    layer_ff_prev->output_layer = layer_object;
    layer_object = prev_layer_object;
  }
  Data_Get_Struct( layer_object, Layer_FF, layer_ff );
  layer_ff->locked_input = 1;

  network->first_layer = layer_object;
  return;
}

int network__count_layers( Network *network ) {
  int count = 0;
  VALUE layer_object;
  Layer_FF *layer_ff;

  layer_object = network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    count++;
    Data_Get_Struct( layer_object, Layer_FF, layer_ff );
    layer_object = layer_ff->output_layer;
  }

  return count;
}

void network__init_layer_weights( Network *network, float min_weight, float max_weight ) {
  VALUE layer_object;
  Layer_FF *layer_ff;

  layer_object = network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, Layer_FF, layer_ff );
    layer_ff__init_weights( layer_ff, min_weight, max_weight );
    layer_object = layer_ff->output_layer;
  }

  return;
}

int network__num_outputs( Network *network ) {
  int count_outputs = 0;
  VALUE layer_object;
  Layer_FF *layer_ff;

  layer_object = network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, Layer_FF, layer_ff );
    count_outputs = layer_ff->num_outputs;
    layer_object = layer_ff->output_layer;
  }

  return count_outputs;
}

int network__num_inputs( Network *network ) {
  Layer_FF *layer_ff;
  Data_Get_Struct( network->first_layer, Layer_FF, layer_ff );
  return layer_ff->num_inputs;
}

// This assumes input has been assigned already to first layer
void network__run( Network *network ) {
  volatile VALUE layer_object;
  Layer_FF *layer_ff;

  layer_object = network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, Layer_FF, layer_ff );
    layer_ff__run( layer_ff );
    layer_object = layer_ff->output_layer;
  }

  return;
}

Layer_FF *network__last_layer_ff( Network *network ) {
  VALUE layer_object;
  Layer_FF *layer_ff;

  layer_object = network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, Layer_FF, layer_ff );
    layer_object = layer_ff->output_layer;
  }

  return layer_ff;
}

void network__calc_output_deltas( Network *network, VALUE val_target ) {
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  struct NARRAY *na_output_slope;
  struct NARRAY *na_output_deltas;
  Layer_FF *layer_ff = network__last_layer_ff( network );

  GetNArray( val_target, na_target );
  GetNArray( layer_ff->narr_output, na_output );
  GetNArray( layer_ff->narr_output_slope, na_output_slope );
  GetNArray( layer_ff->narr_output_deltas, na_output_deltas );

  transfer_bulk_derivative_at( layer_ff->transfer_fn, layer_ff->num_outputs, (float *) na_output->ptr, (float *) na_output_slope->ptr );

  core_calc_output_deltas( layer_ff->num_outputs, (float *) na_output->ptr,
      (float *) na_output_slope->ptr, (float *) na_target->ptr, (float *) na_output_deltas->ptr );

  return;
}

void network__backprop_deltas( Network *network ) {
  Layer_FF *layer_ff_input;
  Layer_FF *layer_ff = network__last_layer_ff( network );
  while ( ! NIL_P(layer_ff->input_layer) ) {
    Data_Get_Struct( layer_ff->input_layer, Layer_FF, layer_ff_input );
    layer_ff__backprop_deltas( layer_ff, layer_ff_input );
    layer_ff = layer_ff_input;
  }
  return;
}

void network__update_weights( Network *network ) {
  Layer_FF *layer_ff;
  Data_Get_Struct( network->first_layer, Layer_FF, layer_ff );

  while ( ! NIL_P(layer_ff->output_layer) ) {
    layer_ff__update_weights( layer_ff, 1.0, 0.5 );
    Data_Get_Struct( layer_ff->output_layer, Layer_FF, layer_ff );
  }
  layer_ff__update_weights( layer_ff, network->eta, network->momentum );
  return;
}

void network__train_once( Network *network, VALUE val_input, VALUE val_target ) {
  Layer_FF *layer_ff;

  ////////////////////////////////////////////////////////////////////////////////////
  // Attach input
  Data_Get_Struct( network->first_layer, Layer_FF, layer_ff );
  layer_ff__set_input( layer_ff, val_input );

  ////////////////////////////////////////////////////////////////////////////////////
  // Run the network forward
  network__run( network );

  ////////////////////////////////////////////////////////////////////////////////////
  // Calculate delta on output
  network__calc_output_deltas( network, val_target );

  ////////////////////////////////////////////////////////////////////////////////////
  // Back-propagate delta
  network__backprop_deltas( network );

  ////////////////////////////////////////////////////////////////////////////////////
  // Adjust weights
  network__update_weights( network );

  return;
}
