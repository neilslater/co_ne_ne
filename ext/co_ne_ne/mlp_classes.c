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
#include "transfer_module.h"

VALUE MLP = Qnil;
VALUE NLayer = Qnil;
VALUE Network = Qnil;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline VALUE mlp_layer_as_ruby_class( MLP_Layer *mlp_layer , VALUE klass ) {
  return Data_Wrap_Struct( klass, mark_mlp_layer_struct, destroy_mlp_layer_struct, mlp_layer );
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

VALUE mlp_layer_class_initialize( int argc, VALUE* argv, VALUE self ) {
  VALUE n_ins, n_outs, tfn_type;
  ID tfn_id;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  int i, o;
  rb_scan_args( argc, argv, "21", &n_ins, &n_outs, &tfn_type );

  i = NUM2INT( n_ins );
  if ( i < 1 ) {
    rb_raise( rb_eArgError, "Input size %d is less than minimum of 1", i );
  }
  o = NUM2INT( n_outs );
  if ( o < 1 ) {
    rb_raise( rb_eArgError, "Output size %d is less than minimum of 1", o );
  }

  mlp_layer->num_inputs = i;
  mlp_layer->num_outputs = o;

  tfn_id = rb_intern("sigmoid");
  if ( ! NIL_P(tfn_type) ) {
    if ( TYPE(tfn_type) != T_SYMBOL ) {
      rb_raise( rb_eTypeError, "Expected symbol for transfer function type" );
    }
    tfn_id = SYM2ID(tfn_type);
  }

  if ( rb_intern("sigmoid") == tfn_id ) {
    mlp_layer->transfer_fn = SIGMOID;
  } else if ( rb_intern("tanh") == tfn_id ) {
     mlp_layer->transfer_fn = TANH;
  } else if ( rb_intern("relu") == tfn_id ) {
     mlp_layer->transfer_fn = RELU;
  } else {
    rb_raise( rb_eArgError, "Transfer function type %s not recognised", rb_id2name(tfn_id) );
  }

  mlp_layer_struct_create_arrays( mlp_layer );
  mlp_layer_struct_init_weights( mlp_layer, -0.8, 0.8 );

  return self;
}

// Special initialize to support "clone"
VALUE mlp_layer_class_initialize_copy( VALUE copy, VALUE orig ) {
  MLP_Layer *mlp_layer_copy;
  MLP_Layer *mlp_layer_orig;

  if (copy == orig) return copy;
  mlp_layer_copy = get_mlp_layer_struct( copy );
  mlp_layer_orig = get_mlp_layer_struct( orig );

  // Should this in fact be a deep clone?
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

VALUE mlp_layer_object_transfer( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  VALUE t;
  switch ( mlp_layer->transfer_fn ) {
    case SIGMOID:
      t = Sigmoid;
      break;
    case TANH:
      t = TanH;
      break;
    case RELU:
      t = ReLU;
      break;
  }
  return t;
}

VALUE mlp_layer_object_input( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_input;
}

VALUE mlp_layer_object_output( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_output;
}

VALUE mlp_layer_object_weights( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_weights;
}

VALUE mlp_layer_object_input_layer( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->input_layer;
}

VALUE mlp_layer_object_output_layer( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->output_layer;
}

VALUE mlp_layer_object_output_deltas( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_output_deltas;
}

VALUE mlp_layer_object_weights_last_deltas( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_weights_last_deltas;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mlp_classes( VALUE parent_module ) {
  // Define modules and classes
  MLP = rb_define_module_under( parent_module, "MLP" );
  NLayer = rb_define_class_under( MLP, "NLayer", rb_cObject );
  Network = rb_define_class_under( MLP, "Network", rb_cObject );

  // NLayer instantiation and class methods
  rb_define_alloc_func( NLayer, mlp_layer_alloc );
  rb_define_method( NLayer, "initialize", mlp_layer_class_initialize, -1 );
  rb_define_method( NLayer, "initialize_copy", mlp_layer_class_initialize_copy, 1 );

  // NLayer attributes
  rb_define_method( NLayer, "num_inputs", mlp_layer_object_num_inputs, 0 );
  rb_define_method( NLayer, "num_outputs", mlp_layer_object_num_outputs, 0 );
  rb_define_method( NLayer, "transfer", mlp_layer_object_transfer, 0 );
  rb_define_method( NLayer, "input", mlp_layer_object_input, 0 );
  rb_define_method( NLayer, "output", mlp_layer_object_output, 0 );
  rb_define_method( NLayer, "weights", mlp_layer_object_weights, 0 );
  rb_define_method( NLayer, "input_layer", mlp_layer_object_input_layer, 0 );
  rb_define_method( NLayer, "output_layer", mlp_layer_object_output_layer, 0 );
  rb_define_method( NLayer, "output_deltas", mlp_layer_object_output_deltas, 0 );
  rb_define_method( NLayer, "weights_last_deltas", mlp_layer_object_weights_last_deltas, 0 );


  // NLayer methods
}
