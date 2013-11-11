// ext/co_ne_ne/mlp_classes.c

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for multi-layer perceptron code - the deeper implementation is in
//  mlp_layer_raw.c and mlp_network_raw.c
//

#include <ruby.h>
#include "narray.h"
#include <stdio.h>
#include <xmmintrin.h>

#include "narray_shared.h"
#include "mlp_classes.h"
#include "mlp_layer_raw.h"

VALUE MLP = Qnil;
VALUE NLayer = Qnil;
VALUE Network = Qnil;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline VALUE mlp_layer_as_ruby_class( MLP_Layer *mlp_layer , VALUE klass ) {
  // TODO: Register mark callback
  return Data_Wrap_Struct( klass, 0, destroy_mlp_layer_struct, mlp_layer );
}

VALUE mlp_layer_alloc(VALUE klass) {
  return mlp_layer_as_ruby_class( create_mlp_layer_struct(), klass );
}

inline MLP_Layer *get_mlp_layer_struct( VALUE obj ) {
  MLP_Layer *mlp_layer;
  Data_Get_Struct( obj, MLP_Layer, mlp_layer );
  return mlp_layer;
}

void assert_value_wraps_mlp_layer( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)destroy_mlp_layer_struct) {
    rb_raise( rb_eTypeError, "Expected a NLayer object, but got something else" );
  }
}

//////////////////////////////////////////////////////////////////////////////////////
//
//  NLayer method definitions
//

// Native extensions version of initialize
VALUE mlp_layer_class_initialize( VALUE self, VALUE n_ins, VALUE n_outs ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );


  mlp_layer->num_inputs = NUM2INT( n_ins );
  mlp_layer->num_outputs = NUM2INT( n_outs );

  return self;
}

// Special initialize to support "clone"
VALUE mlp_layer_class_initialize_copy( VALUE copy, VALUE orig ) {
  MLP_Layer *mlp_layer_copy;
  MLP_Layer *mlp_layer_orig;

  if (copy == orig) return copy;
  mlp_layer_copy = get_mlp_layer_struct( copy );
  mlp_layer_orig = get_mlp_layer_struct( orig );
  memcpy( mlp_layer_copy, mlp_layer_orig, sizeof(MLP_Layer) );

  return copy;
}

VALUE mlp_layer_object_num_inputs( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return INT2FIX( mlp_layer->num_inputs );
}

VALUE mlp_layer_object_num_outputs( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return INT2FIX( mlp_layer->num_outputs );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mlp_classes( VALUE parent_module ) {
  MLP = rb_define_module_under( parent_module, "MLP" );

  NLayer = rb_define_class_under( MLP, "NLayer", rb_cObject );
  rb_define_alloc_func( NLayer, mlp_layer_alloc );

  // NLayer instantiation and class methods
  rb_define_method( NLayer, "initialize", mlp_layer_class_initialize, 2 );
  rb_define_method( NLayer, "initialize_copy", mlp_layer_class_initialize_copy, 1 );

  // NLayer attributes
  rb_define_method( NLayer, "num_inputs", mlp_layer_object_num_inputs, 0 );
  rb_define_method( NLayer, "num_outputs", mlp_layer_object_num_outputs, 0 );

  // NLayer methods

  Network = rb_define_class_under( MLP, "Network", rb_cObject );
}
