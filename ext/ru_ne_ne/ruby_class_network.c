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

/* @overload initialize( nn_model, learn )
 * Creates a new ...
 * @param [Object] nn_model ...
 * @param [Object] learn ...
 * @return [RuNeNe::Network] new ...
 */
VALUE network_rbobject__initialize( VALUE self, VALUE rv_nn_model, VALUE rv_learn ) {
  Network *network = get_network_struct( self );

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

/* @!attribute [r] nn_model
 * Description goes here
 * @return [Object]
 */
VALUE network_rbobject__get_nn_model( VALUE self ) {
  Network *network = get_network_struct( self );
  return network->nn_model;
}

/* @!attribute [r] learn
 * Description goes here
 * @return [Object]
 */
VALUE network_rbobject__get_learn( VALUE self ) {
  Network *network = get_network_struct( self );
  return network->learn;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_network_class( ) {
  // Network instantiation and class methods
  rb_define_alloc_func( RuNeNe_Network, network_alloc );
  rb_define_method( RuNeNe_Network, "initialize", network_rbobject__initialize, 2 );
  rb_define_method( RuNeNe_Network, "initialize_copy", network_rbobject__initialize_copy, 1 );

  // Network attributes
  rb_define_method( RuNeNe_Network, "nn_model", network_rbobject__get_nn_model, 0 );
  rb_define_method( RuNeNe_Network, "learn", network_rbobject__get_learn, 0 );
}
