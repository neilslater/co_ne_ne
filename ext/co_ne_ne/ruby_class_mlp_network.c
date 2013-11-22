// ext/co_ne_ne/ruby_class_mlp_network.c

#include "ruby_class_mlp_network.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for multi-layer perceptron code - the deeper implementation is in
//  struct_mlp_layer.c and struct_mlp_network.c
//

inline VALUE mlp_network_as_ruby_class( MLP_Network *mlp_network , VALUE klass ) {
  return Data_Wrap_Struct( klass, p_mlp_network_gc_mark, p_mlp_network_destroy, mlp_network );
}

VALUE mlp_network_alloc(VALUE klass) {
  return mlp_network_as_ruby_class( p_mlp_network_create(), klass );
}

inline MLP_Network *get_mlp_network_struct( VALUE obj ) {
  MLP_Network *mlp_network;
  Data_Get_Struct( obj, MLP_Network, mlp_network );
  return mlp_network;
}

void assert_value_wraps_mlp_network( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)p_mlp_network_destroy) {
    rb_raise( rb_eTypeError, "Expected a Network object, but got something else" );
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Layer method definitions
//

VALUE mlp_network_class_initialize( VALUE self, VALUE num_inputs, VALUE hidden_layers, VALUE num_outputs ) {
  int ninputs, noutputs, i, nhlayers, hlsize, *layer_sizes;
  MLP_Network *mlp_network = get_mlp_network_struct( self );
  ninputs = NUM2INT( num_inputs );
  noutputs = NUM2INT( num_outputs );

  if (ninputs < 1) {
    rb_raise( rb_eArgError, "Input size %d not allowed.", ninputs );
  }

  if (noutputs < 1) {
    rb_raise( rb_eArgError, "Output size %d not allowed.", noutputs );
  }

  // Pre-check all array entries before initialising further
  Check_Type( hidden_layers, T_ARRAY );
  nhlayers = FIX2INT( rb_funcall( hidden_layers, rb_intern("count"), 0 ) );
  for ( i = 0; i < nhlayers; i++ ) {
    hlsize = FIX2INT( rb_ary_entry( hidden_layers, i ) );
    if ( hlsize < 1 ) {
      rb_raise( rb_eArgError, "Hidden layer output size %d not allowed.", hlsize );
    }
  }

  layer_sizes = ALLOC_N( int, nhlayers + 2 );
  layer_sizes[0] = ninputs;
  for ( i = 0; i < nhlayers; i++ ) {
    layer_sizes[i+1] = FIX2INT( rb_ary_entry( hidden_layers, i ) );
  }
  layer_sizes[nhlayers+1] = noutputs;

  p_mlp_network_init_layers( mlp_network, nhlayers + 1, layer_sizes );

  xfree( layer_sizes );
  return self;
}

// Special initialize to support "clone"
VALUE mlp_network_class_initialize_copy( VALUE copy, VALUE orig ) {
  rb_raise( rb_eArgError, "Cannot clone CoNeNe::MLP::Network (yet)." );
  return copy;
}

VALUE mlp_network_object_num_layers( VALUE self ) {
  int count = 0;
  VALUE layer_object;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    count++;
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  return INT2NUM( count );
}

VALUE mlp_network_object_layers( VALUE self ) {
  int count = 0;
  VALUE layer_object, all_layers;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    count++;
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  all_layers = rb_ary_new2( count );
  count = 0;
  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    rb_ary_store( all_layers, count, layer_object );
    count++;
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  return all_layers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mlp_network_class( VALUE parent_module ) {
  // Layer instantiation and class methods
  rb_define_alloc_func( Network, mlp_network_alloc );
  rb_define_method( Network, "initialize", mlp_network_class_initialize, 3 );
  rb_define_method( Network, "initialize_copy", mlp_network_class_initialize_copy, 1 );

  // Network attributes
  rb_define_method( Network, "num_layers", mlp_network_object_num_layers, 0 );
  rb_define_method( Network, "layers", mlp_network_object_layers, 0 );

  // Network methods

}
