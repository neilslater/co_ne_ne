// ext/ru_ne_ne/ruby_class_network.c

#include "ruby_class_network.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_network.c
//

inline VALUE network_as_ruby_class( Network *network , VALUE klass ) {
  return Data_Wrap_Struct( klass, network__gc_mark, network__destroy, network );
}

VALUE network_alloc(VALUE klass) {
  return network_as_ruby_class( network__create(), klass );
}

inline Network *get_network_struct( VALUE obj ) {
  Network *network;
  Data_Get_Struct( obj, Network, network );
  return network;
}

void assert_value_wraps_network( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)network__destroy) {
    rb_raise( rb_eTypeError, "Expected a Network object, but got something else" );
  }
}

/* Document-class: RuNeNe::Network
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Network method definitions
//

/* @overload initialize( layers )
 * Creates a new ...
 * @param [Array<RuNeNe::Layer::Feedforward>] layers ...
 * @return [RuNeNe::Network] new ...
 */
VALUE network_rbobject__initialize( VALUE self, VALUE rv_layers ) {
  Network *network = get_network_struct( self );
  // This stack-based var avoids memory leaks from alloc which might not be freed on error
  VALUE layers[100];
  volatile VALUE current_layer;
  int i, n;

  Check_Type( rv_layers, T_ARRAY );

  n = FIX2INT( rb_funcall( rv_layers, rb_intern("count"), 0 ) );
  if ( n < 1 ) {
    rb_raise( rb_eArgError, "no layers in network" );
  }
  if ( n > 100 ) {
    rb_raise( rb_eArgError, "too many layers in network" );
  }

  for ( i = 0; i < n; i++ ) {
    current_layer = rb_ary_entry( rv_layers, i );
    // TODO: Accept more than one definition of layer (e.g. orig object, hash). Support
    //       multiple layer types in theory.
    assert_value_wraps_layer_ff( current_layer );
    layers[i] = current_layer;
  }

  network__init( network, n, layers);

  return self;
}

/* @overload clone
 * When cloned, the returned Network has deep copies of C data.
 * @return [RuNeNe::Network] new
 */
VALUE network_rbobject__initialize_copy( VALUE copy, VALUE orig ) {
  Network *network_copy;
  Network *network_orig;

  if (copy == orig) return copy;
  network_orig = get_network_struct( orig );
  network_copy = get_network_struct( copy );

  network__deep_copy( network_copy, network_orig );

  return copy;
}

/* @!attribute [r] layers
 * Description goes here
 * @return [Array<RuNeNe::Layer::Feedforward>]]
 */
VALUE network_rbobject__get_layers( VALUE self ) {
  Network *network = get_network_struct( self );
  int i;

  volatile VALUE rv_layers = rb_ary_new2( network->num_layers );
  for ( i = 0; i < network->num_layers; i++ ) {
    rb_ary_store( rv_layers, i, network->layers[i] );
  }

  return rv_layers;
}

/* @!attribute [r] num_layers
 * Description goes here
 * @return [Integer]
 */
VALUE network_rbobject__get_num_layers( VALUE self ) {
  Network *network = get_network_struct( self );
  return INT2NUM( network->num_layers );
}

/* @!attribute [r] num_inputs
 * Description goes here
 * @return [Integer]
 */
VALUE network_rbobject__get_num_inputs( VALUE self ) {
  Network *network = get_network_struct( self );
  return INT2NUM( network->num_inputs );
}

/* @!attribute [r] num_outputs
 * Description goes here
 * @return [Integer]
 */
VALUE network_rbobject__get_num_outputs( VALUE self ) {
  Network *network = get_network_struct( self );
  return INT2NUM( network->num_outputs );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_network_class( ) {
  // Network instantiation and class methods
  rb_define_alloc_func( RuNeNe_Network, network_alloc );
  rb_define_method( RuNeNe_Network, "initialize", network_rbobject__initialize, 1 );
  rb_define_method( RuNeNe_Network, "initialize_copy", network_rbobject__initialize_copy, 1 );

  // Network attributes
  rb_define_method( RuNeNe_Network, "layers", network_rbobject__get_layers, 0 );
  rb_define_method( RuNeNe_Network, "num_layers", network_rbobject__get_num_layers, 0 );
  rb_define_method( RuNeNe_Network, "num_inputs", network_rbobject__get_num_inputs, 0 );
  rb_define_method( RuNeNe_Network, "num_outputs", network_rbobject__get_num_outputs, 0 );
}
