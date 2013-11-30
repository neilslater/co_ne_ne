// ext/co_ne_ne/ruby_class_mlp_layer.c

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for multi-layer perceptron code - the deeper implementation is in
//  mlp_layer_raw.c and mlp_network_raw.c
//

#include "ruby_class_mlp_layer.h"

// These are extern in other code (TODO: Move module and class defs to co_ne_ne.c ?)
VALUE MLP = Qnil;
VALUE Layer = Qnil;
VALUE Network = Qnil;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline VALUE mlp_layer_as_ruby_class( MLP_Layer *mlp_layer, VALUE klass ) {
  return Data_Wrap_Struct( klass, p_mlp_layer_gc_mark, p_mlp_layer_destroy, mlp_layer );
}

VALUE mlp_layer_alloc(VALUE klass) {
  return mlp_layer_as_ruby_class( p_mlp_layer_create(), klass );
}


inline MLP_Layer *get_mlp_layer_struct( VALUE obj ) {
  MLP_Layer *mlp_layer;
  Data_Get_Struct( obj, MLP_Layer, mlp_layer );
  return mlp_layer;
}

VALUE mlp_layer_new_ruby_object( int n_inputs, int n_outputs, transfer_type tfn ) {
  MLP_Layer *mlp_layer;
  VALUE mlp_layer_ruby = mlp_layer_alloc( Layer );
  mlp_layer = get_mlp_layer_struct( mlp_layer_ruby );

  mlp_layer->num_inputs = n_inputs;
  mlp_layer->num_outputs = n_outputs;
  mlp_layer->transfer_fn = tfn;

  p_mlp_layer_new_narrays( mlp_layer );
  p_mlp_layer_init_weights( mlp_layer, -0.8, 0.8 );

  return mlp_layer_ruby;
}

VALUE mlp_layer_clone_ruby_object( VALUE orig ) {
  volatile VALUE copy;
  MLP_Layer *mlp_layer_copy;
  MLP_Layer *mlp_layer_orig;
  mlp_layer_orig = get_mlp_layer_struct( orig );

  copy =  mlp_layer_alloc( Layer );
  mlp_layer_copy = get_mlp_layer_struct( copy );

  mlp_layer_copy->num_inputs = mlp_layer_orig->num_inputs;
  mlp_layer_copy->num_outputs = mlp_layer_orig->num_outputs;
  mlp_layer_copy->transfer_fn = mlp_layer_orig->transfer_fn;

  mlp_layer_copy->narr_input = Qnil;
  mlp_layer_copy->input_layer = Qnil;
  mlp_layer_copy->output_layer = Qnil;

  mlp_layer_copy->narr_output = na_clone( mlp_layer_orig->narr_output );
  mlp_layer_copy->narr_weights = na_clone( mlp_layer_orig->narr_weights );
  mlp_layer_copy->narr_output_deltas = na_clone( mlp_layer_orig->narr_output_deltas );
  mlp_layer_copy->narr_weights_last_deltas = na_clone( mlp_layer_orig->narr_weights_last_deltas );
  mlp_layer_copy->narr_output_slope = na_clone( mlp_layer_orig->narr_output_slope );

  return copy;
}

void assert_value_wraps_mlp_layer( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)p_mlp_layer_destroy) {
    rb_raise( rb_eTypeError, "Expected a Layer object, but got something else" );
  }
}

transfer_type transfer_fn_from_symbol( VALUE tfn_type ) {
  ID tfn_id;

  tfn_id = rb_intern("sigmoid");
  if ( ! NIL_P(tfn_type) ) {
    if ( TYPE(tfn_type) != T_SYMBOL ) {
      rb_raise( rb_eTypeError, "Expected symbol for transfer function type" );
    }
    tfn_id = SYM2ID(tfn_type);
  }

  if ( rb_intern("sigmoid") == tfn_id ) {
    return SIGMOID;
  } else if ( rb_intern("tanh") == tfn_id ) {
    return TANH;
  } else if ( rb_intern("relu") == tfn_id ) {
    return RELU;
  } else {
    rb_raise( rb_eArgError, "Transfer function type %s not recognised", rb_id2name(tfn_id) );
  }
}

void set_transfer_fn_from_symbol( MLP_Layer *mlp_layer , VALUE tfn_type ) {
  mlp_layer->transfer_fn = transfer_fn_from_symbol( tfn_type );
}

void assert_not_in_output_chain( MLP_Layer *mlp_layer, VALUE unexpected_layer ) {
  MLP_Layer *mlp_next_layer = mlp_layer;
  while ( ! NIL_P( mlp_next_layer->output_layer ) ) {
    if ( mlp_next_layer->output_layer == unexpected_layer ) {
      rb_raise( rb_eArgError, "Attempt to create a circular network." );
    }
    mlp_next_layer = get_mlp_layer_struct( mlp_next_layer->output_layer );
  }
  return;
}

void assert_not_in_input_chain( MLP_Layer *mlp_layer, VALUE unexpected_layer ) {
  MLP_Layer *mlp_next_layer = mlp_layer;
  while ( ! NIL_P( mlp_next_layer->input_layer ) ) {
    if ( mlp_next_layer->input_layer == unexpected_layer ) {
      rb_raise( rb_eArgError, "Attempt to create a circular network." );
    }
    mlp_next_layer = get_mlp_layer_struct( mlp_next_layer->input_layer );
  }
  return;
}

VALUE mlp_layer_new_ruby_object_from_weights( VALUE weights, transfer_type tfn ) {
  MLP_Layer *mlp_layer;
  struct NARRAY *na_weights;
  VALUE mlp_layer_ruby = mlp_layer_alloc( Layer );
  mlp_layer = get_mlp_layer_struct( mlp_layer_ruby );

  GetNArray( weights, na_weights );
  mlp_layer->num_inputs = na_weights->shape[0] - 1;
  mlp_layer->num_outputs = na_weights->shape[1];
  mlp_layer->transfer_fn = tfn;
  p_mlp_layer_init_from_weights( mlp_layer, weights );

  return mlp_layer_ruby;
}


/* Document-module:  CoNeNe::MLP::Layer
 *
 * An object of this class represents a layer in a fully connected feed-forward network,
 * with inputs, weights and outputs. The inputs and outputs may be shared with other
 * layers by attaching the layers together. This can be done at any time.
 *
 * A layer may only be connected to a single input and output at any one time. Making a
 * new attachment will remove existing attachments that conflict. The first layer in a network
 * may not have a new input layer attached.
 */

//////////////////////////////////////////////////////////////////////////////////////
//
//  Layer method definitions
//

/* @overload initialize( num_inputs, num_outputs, transfer_label = :sigmoid )
 * Creates a new layer and randomly initializes the weights (between -0.8 and 0.8).
 * @param [Integer] num_inputs size of input array
 * @param [Integer] num_outputs size of output array
 * @param [Symbol] transfer_label type of transfer function to use.
 * @return [CoNeNe::MLP::Layer] new layer with random weights.
 */
VALUE mlp_layer_class_initialize( int argc, VALUE* argv, VALUE self ) {
  VALUE n_ins, n_outs, tfn_type;
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

  set_transfer_fn_from_symbol( mlp_layer, tfn_type );

  p_mlp_layer_new_narrays( mlp_layer );
  p_mlp_layer_init_weights( mlp_layer, -0.8, 0.8 );

  return self;
}

// Special initialize to support "clone"
VALUE mlp_layer_class_initialize_copy( VALUE copy, VALUE orig ) {
  MLP_Layer *mlp_layer_copy;
  MLP_Layer *mlp_layer_orig;

  if (copy == orig) return copy;
  mlp_layer_copy = get_mlp_layer_struct( copy );
  mlp_layer_orig = get_mlp_layer_struct( orig );

  mlp_layer_copy->num_inputs = mlp_layer_orig->num_inputs;
  mlp_layer_copy->num_outputs = mlp_layer_orig->num_outputs;
  mlp_layer_copy->transfer_fn = mlp_layer_orig->transfer_fn;

  mlp_layer_copy->narr_input = Qnil;
  mlp_layer_copy->input_layer = Qnil;
  mlp_layer_copy->output_layer = Qnil;

  mlp_layer_copy->narr_output = na_clone( mlp_layer_orig->narr_output );
  mlp_layer_copy->narr_weights = na_clone( mlp_layer_orig->narr_weights );
  mlp_layer_copy->narr_output_deltas = na_clone( mlp_layer_orig->narr_output_deltas );
  mlp_layer_copy->narr_weights_last_deltas = na_clone( mlp_layer_orig->narr_weights_last_deltas );
  mlp_layer_copy->narr_output_slope = na_clone( mlp_layer_orig->narr_output_slope );

  return copy;
}

VALUE mlp_layer_class_from_weights( int argc, VALUE* argv, VALUE self ) {
  VALUE weights_in, tfn_type;
  struct NARRAY *na_weights;
  volatile VALUE val_weights;
  int i, o;

  rb_scan_args( argc, argv, "11", &weights_in, &tfn_type );

  val_weights = na_cast_object(weights_in, NA_SFLOAT);
  GetNArray( val_weights, na_weights );

  if ( na_weights->rank != 2 ) {
    rb_raise( rb_eArgError, "Weights rank should be 2, but got %d", na_weights->rank );
  }

  i = na_weights->shape[0] - 1;
  if ( i < 1 ) {
    rb_raise( rb_eArgError, "Input size %d is less than minimum of 1", i );
  }
  o = na_weights->shape[1];
  if ( o < 1 ) {
    rb_raise( rb_eArgError, "Output size %d is less than minimum of 1", o );
  }

  return mlp_layer_new_ruby_object_from_weights( val_weights, transfer_fn_from_symbol( tfn_type ) );
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

VALUE mlp_layer_object_init_weights( int argc, VALUE* argv, VALUE self ) {
  VALUE minw, maxw;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  float min_weight, max_weight;
  rb_scan_args( argc, argv, "02", &minw, &maxw );

  if ( ! NIL_P(minw) ) {
    min_weight = NUM2FLT( minw );
    if ( ! NIL_P(maxw) ) {
      max_weight = NUM2FLT( maxw );
    } else {
      max_weight = min_weight;
      min_weight = -max_weight;
    }
  } else {
    min_weight = -0.8;
    max_weight = 0.8;
  }

  p_mlp_layer_init_weights( mlp_layer, min_weight, max_weight );

  return Qnil;
}

VALUE mlp_layer_object_set_input( VALUE self, VALUE new_input ) {
  struct NARRAY *na_input;
  volatile VALUE val_input;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );

  val_input = na_cast_object(new_input, NA_SFLOAT);
  GetNArray( val_input, na_input );

  if ( na_input->rank != 1 ) {
    rb_raise( rb_eArgError, "Inputs rank should be 1, but got %d", na_input->rank );
  }

  if ( na_input->total != mlp_layer->num_inputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer input size %d", na_input->total, mlp_layer->num_inputs );
  }

  p_mlp_layer_set_input( mlp_layer, val_input );

  return val_input;
}

VALUE mlp_layer_object_attach_input_layer( VALUE self, VALUE new_input_layer ) {
  MLP_Layer *mlp_new_input_layer;
  MLP_Layer *mlp_old_output_layer, *mlp_old_input_layer;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );

  if ( mlp_layer->locked_input > 0 ) {
    rb_raise( rb_eArgError, "Layer has been marked as 'first layer' and may not have another input layer attached." );
  }

  assert_value_wraps_mlp_layer( new_input_layer );
  mlp_new_input_layer = get_mlp_layer_struct( new_input_layer );

  if ( mlp_new_input_layer->num_outputs != mlp_layer->num_inputs ) {
    rb_raise( rb_eArgError, "Input layer output size %d does not match layer input size %d", mlp_new_input_layer->num_outputs, mlp_layer->num_inputs );
  }

  assert_not_in_input_chain( mlp_new_input_layer, self );

  if ( ! NIL_P( mlp_layer->input_layer ) ) {
    // This layer has an existing input layer, it needs to stop pointing its output here
    mlp_old_input_layer = get_mlp_layer_struct( mlp_layer->input_layer );
    mlp_old_input_layer->output_layer = Qnil;
  }

  mlp_layer->narr_input = mlp_new_input_layer->narr_output;
  mlp_layer->input_layer = new_input_layer;

  if ( ! NIL_P( mlp_new_input_layer->output_layer ) ) {
    // The new input layer was previously attached elsewhere. This needs to be disconnected too
    mlp_old_output_layer = get_mlp_layer_struct( mlp_new_input_layer->output_layer );
    mlp_old_output_layer->narr_input = Qnil;
    mlp_old_output_layer->input_layer = Qnil;
  }
  mlp_new_input_layer->output_layer = self;

  return new_input_layer;
}


VALUE mlp_layer_object_attach_output_layer( VALUE self, VALUE new_output_layer ) {
  MLP_Layer *mlp_new_output_layer;
  MLP_Layer *mlp_old_output_layer, *mlp_old_input_layer;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );

  assert_value_wraps_mlp_layer( new_output_layer );
  mlp_new_output_layer = get_mlp_layer_struct( new_output_layer );

  if ( mlp_new_output_layer->locked_input > 0 ) {
    rb_raise( rb_eArgError, "Target layer has been marked as 'first layer' and may not have another input layer attached." );
  }

  if ( mlp_new_output_layer->num_inputs != mlp_layer->num_outputs ) {
    rb_raise( rb_eArgError, "Output layer input size %d does not match layer output size %d", mlp_new_output_layer->num_inputs, mlp_layer->num_outputs );
  }

  assert_not_in_output_chain( mlp_new_output_layer, self );

  if ( ! NIL_P( mlp_layer->output_layer ) ) {
    // This layer has an existing output layer, it needs to stop pointing its input here
    mlp_old_output_layer = get_mlp_layer_struct( mlp_layer->output_layer );
    mlp_old_output_layer->input_layer = Qnil;
    mlp_old_output_layer->narr_input = Qnil;
  }

  mlp_layer->output_layer = new_output_layer;

  if ( ! NIL_P( mlp_new_output_layer->input_layer ) ) {
    // The new output layer was previously attached elsewhere. This needs to be disconnected too
    mlp_old_input_layer = get_mlp_layer_struct( mlp_new_output_layer->input_layer );
    mlp_old_input_layer->output_layer = Qnil;
  }
  mlp_new_output_layer->input_layer = self;
  mlp_new_output_layer->narr_input = mlp_layer->narr_output;

  return new_output_layer;
}

VALUE mlp_layer_object_run( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );

  if ( NIL_P( mlp_layer->narr_input ) ) {
    rb_raise( rb_eArgError, "No input. Cannot run MLP layer." );
  }

  p_mlp_layer_run( mlp_layer );

  return Qnil;
}

VALUE mlp_layer_object_ms_error( VALUE self, VALUE target ) {
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  volatile VALUE val_target;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );

  val_target = na_cast_object(target, NA_SFLOAT);
  GetNArray( val_target, na_target );

  if ( na_target->rank != 1 ) {
    rb_raise( rb_eArgError, "Target output rank should be 1, but got %d", na_target->rank );
  }

  if ( na_target->total != mlp_layer->num_outputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer output size %d", na_target->total, mlp_layer->num_outputs );
  }

  GetNArray( mlp_layer->narr_output, na_output );

  return FLT2NUM( core_mean_square_error( mlp_layer->num_outputs, (float *) na_output->ptr,  (float *) na_target->ptr ) );
}

VALUE mlp_layer_object_calc_output_deltas( VALUE self, VALUE target ) {
  struct NARRAY *na_target;
  volatile VALUE val_target;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );

  val_target = na_cast_object(target, NA_SFLOAT);
  GetNArray( val_target, na_target );

  if ( na_target->rank != 1 ) {
    rb_raise( rb_eArgError, "Target output rank should be 1, but got %d", na_target->rank );
  }

  if ( na_target->total != mlp_layer->num_outputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer output size %d", na_target->total, mlp_layer->num_outputs );
  }

  p_mlp_layer_calc_output_deltas( mlp_layer, val_target );

  return mlp_layer->narr_output_deltas;
}

VALUE mlp_layer_object_backprop_deltas( VALUE self ) {
  MLP_Layer *mlp_layer_input;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );

  if ( NIL_P( mlp_layer->input_layer ) ) {
    rb_raise( rb_eArgError, "No input layer. Cannot run MLP backpropagation." );
  }

  mlp_layer_input = get_mlp_layer_struct( mlp_layer->input_layer );

  p_mlp_layer_backprop_deltas( mlp_layer, mlp_layer_input );

  return mlp_layer_input->narr_output_deltas;
}

VALUE mlp_layer_object_update_weights( int argc, VALUE* argv, VALUE self ) {
  VALUE learning_rate, momentum;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  float eta, m;
  rb_scan_args( argc, argv, "11", &learning_rate, &momentum );

  eta = NUM2FLT( learning_rate );
  if ( eta < 0.0 || eta > 10.0 ) {
    rb_raise( rb_eArgError, "Learning rate %0.6f out of bounds (0.0 to 10.0).", eta );
  }

  m = 0.0;

  if ( ! NIL_P(momentum) ) {
    m = NUM2FLT( momentum );
    if ( m < 0.0 || m > 0.95 ) {
      rb_raise( rb_eArgError, "Momentum %0.6f out of bounds (0.0 to 0.95).", m );
    }
  }

  p_mlp_layer_update_weights( mlp_layer, eta, m );

  return Qnil;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mlp_layer_class( ) {
  volatile VALUE conene_root = rb_define_module( "CoNeNe" );

  // Define modules and classes
  MLP = rb_define_module_under( conene_root, "MLP" );
  Layer = rb_define_class_under( MLP, "Layer", rb_cObject );
  Network = rb_define_class_under( MLP, "Network", rb_cObject );

  // Layer instantiation and class methods
  rb_define_alloc_func( Layer, mlp_layer_alloc );
  rb_define_method( Layer, "initialize", mlp_layer_class_initialize, -1 );
  rb_define_method( Layer, "initialize_copy", mlp_layer_class_initialize_copy, 1 );
  rb_define_singleton_method( Layer, "from_weights", mlp_layer_class_from_weights, -1 );

  // Layer attributes
  rb_define_method( Layer, "num_inputs", mlp_layer_object_num_inputs, 0 );
  rb_define_method( Layer, "num_outputs", mlp_layer_object_num_outputs, 0 );
  rb_define_method( Layer, "transfer", mlp_layer_object_transfer, 0 );
  rb_define_method( Layer, "input", mlp_layer_object_input, 0 );
  rb_define_method( Layer, "output", mlp_layer_object_output, 0 );
  rb_define_method( Layer, "weights", mlp_layer_object_weights, 0 );
  rb_define_method( Layer, "input_layer", mlp_layer_object_input_layer, 0 );
  rb_define_method( Layer, "output_layer", mlp_layer_object_output_layer, 0 );
  rb_define_method( Layer, "output_deltas", mlp_layer_object_output_deltas, 0 );
  rb_define_method( Layer, "weights_last_deltas", mlp_layer_object_weights_last_deltas, 0 );

  // Layer methods
  rb_define_method( Layer, "init_weights", mlp_layer_object_init_weights, -1 );
  rb_define_method( Layer, "set_input", mlp_layer_object_set_input, 1 );
  rb_define_method( Layer, "attach_input_layer", mlp_layer_object_attach_input_layer, 1 );
  rb_define_method( Layer, "attach_output_layer", mlp_layer_object_attach_output_layer, 1 );
  rb_define_method( Layer, "run", mlp_layer_object_run, 0 );
  rb_define_method( Layer, "ms_error", mlp_layer_object_ms_error, 1 );
  rb_define_method( Layer, "calc_output_deltas", mlp_layer_object_calc_output_deltas, 1 );
  rb_define_method( Layer, "backprop_deltas", mlp_layer_object_backprop_deltas, 0 );
  rb_define_method( Layer, "update_weights", mlp_layer_object_update_weights, -1 );
}
