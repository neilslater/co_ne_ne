// ext/co_ne_ne/ruby_class_net_training.c

#include "ruby_class_net_training.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for multi-layer perceptron code - the deeper implementation is in
//  struct_mlp_layer.c and struct_net_training.c
//

inline VALUE net_training_as_ruby_class( NetTraining *net_training , VALUE klass ) {
  return Data_Wrap_Struct( klass, p_net_training_gc_mark, p_net_training_destroy, net_training );
}

VALUE net_training_alloc(VALUE klass) {
  return net_training_as_ruby_class( p_net_training_create(), klass );
}

inline NetTraining *get_net_training_struct( VALUE obj ) {
  NetTraining *net_training;
  Data_Get_Struct( obj, NetTraining, net_training );
  return net_training;
}

void assert_value_wraps_net_training( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)p_net_training_destroy) {
    rb_raise( rb_eTypeError, "Expected a Network object, but got something else" );
  }
}

/* Document-class:  CoNeNe::Net::Training
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Network method definitions
//

/* @overload initialize( inputs, targets )
 * Creates a new network and initializes the weights in all layers.
 * @param [NArray] inputs size of input array for first layer
 * @param [NArray] targets sizes of output arrays for each hidden layer
 * @return [CoNeNe::Net::Training] new network consisting of new layers, with random weights
 */
VALUE net_training_class_initialize( VALUE self, VALUE inputs, VALUE targets ) {
  NetTraining *net_training = get_net_training_struct( self );
  return self;
}

/* @overload clone
 * When cloned, the new Network has deep copies of all layers (which in
 * turn have deep copies of all weights etc)
 * @return [CoNeNe::MLP::Network] new network with same weights.
 */
VALUE net_training_class_initialize_copy( VALUE copy, VALUE orig ) {
  NetTraining *net_training_copy;
  NetTraining *net_training_orig;
  return copy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_net_training_class( ) {
  volatile VALUE net_module;
  volatile VALUE training_class;
  volatile VALUE conene_root = rb_define_module( "CoNeNe" );

  // These temporary pointers are necessary for YARD to find the right names
  net_module = rb_define_module_under( conene_root, "Net" );
  training_class = rb_define_class_under( net_module, "Training", rb_cObject );

  // Network instantiation and class methods
  rb_define_alloc_func( training_class, net_training_alloc );
  rb_define_method( training_class, "initialize", net_training_class_initialize, 3 );
}
