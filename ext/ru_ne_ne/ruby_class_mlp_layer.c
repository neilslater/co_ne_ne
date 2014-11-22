// ext/ru_ne_ne/ruby_class_mlp_layer.c

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for multi-layer perceptron code - the deeper implementation is in
//  mlp_layer_raw.c and mlp_network_raw.c
//

#include "ruby_class_mlp_layer.h"

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
  VALUE mlp_layer_ruby = mlp_layer_alloc( RuNeNe_Layer_FeedForward );
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

  copy =  mlp_layer_alloc( RuNeNe_Layer_FeedForward );
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
  } else if ( rb_intern("linear") == tfn_id ) {
    return LINEAR;
  } else if ( rb_intern("softmax") == tfn_id ) {
    return SOFTMAX;
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
  VALUE mlp_layer_ruby = mlp_layer_alloc( RuNeNe_Layer_FeedForward );
  mlp_layer = get_mlp_layer_struct( mlp_layer_ruby );

  GetNArray( weights, na_weights );
  mlp_layer->num_inputs = na_weights->shape[0] - 1;
  mlp_layer->num_outputs = na_weights->shape[1];
  mlp_layer->transfer_fn = tfn;
  p_mlp_layer_init_from_weights( mlp_layer, weights );

  return mlp_layer_ruby;
}


/* Document-class:  RuNeNe::Layer::FeedForward
 *
 * An object of this class represents a layer in a fully connected feed-forward network,
 * with inputs, weights and outputs. The inputs and outputs may be shared with other
 * layers by attaching the layers together. This can be done at any time.
 *
 * A layer may only be connected to a single input and output at any one time. Making a
 * new attachment will remove existing attachments that conflict. The first layer in a network
 * may not have a new input layer attached.
 *
 * A general rule for using NArray parameters with this class is that *sfloat* NArrays
 * are used directly, and other types are cast to that type. This means that using
 * *sfloat* sub-type to manage input data and weights is generally more efficient.
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
 * @return [RuNeNe::Layer::FeedForward] new layer with random weights.
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

/* @overload clone
 * When cloned, the returned Layer has deep copies of weights and outputs,
 * and is *not* connected to the inputs and outputs that the original was.
 * @return [RuNeNe::Layer::FeedForward] new layer same weights and transfer function.
 */
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

/* @overload from_weights( weights, transfer_label = :sigmoid )
 * Creates a new layer using the supplied weights array, which must be rank 2.
 * The inputs and bias are taken from the first dimension, and each output is assigned
 * from the second dimension. For example an array with shape [5,3] has 4 inputs and
 * 3 outputs.
 * @param [NArray] weights
 * @param [Symbol] transfer_label type of transfer function to use.
 * @return [RuNeNe::Layer::FeedForward] new layer using supplied weights.
 */
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

/* @!attribute [r] num_inputs
 * Number of inputs to the layer. This affects the size of arrays when setting the input.
 * @return [Integer]
 */
VALUE mlp_layer_object_num_inputs( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return INT2FIX( mlp_layer->num_inputs );
}

/* @!attribute [r] num_outputs
 * Number of outputs from the layer. This affects the size of arrays for training targets.
 * @return [Integer]
 */
VALUE mlp_layer_object_num_outputs( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return INT2FIX( mlp_layer->num_outputs );
}

/* @!attribute [r] transfer
 * The RuNeNe::Transfer *Module* that is used for transfer methods when the layer is #run.
 * @return [Module]
 */
VALUE mlp_layer_object_transfer( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  VALUE t;
  switch ( mlp_layer->transfer_fn ) {
    case SIGMOID:
      t = RuNeNe_Transfer_Sigmoid;
      break;
    case TANH:
      t = RuNeNe_Transfer_TanH;
      break;
    case RELU:
      t = RuNeNe_Transfer_ReLU;
      break;
    case LINEAR:
      t = RuNeNe_Transfer_Linear;
      break;
    case SOFTMAX:
      t = RuNeNe_Transfer_Softmax;
      break;
  }
  return t;
}

/* @!attribute [r] input
 * The current input array. If there is another layer attached to the input, this will be the
 * same object as the input layer's #output.
 * @return [NArray<sfloat>,nil] one-dimensional array of #num_inputs single-precision floats
 */
VALUE mlp_layer_object_input( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_input;
}

/* @!attribute [r] output
 * The current output array.
 * @return [NArray<sfloat>] one-dimensional array of #num_outputs single-precision floats
 */
VALUE mlp_layer_object_output( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_output;
}

/* @!attribute [r] weights
 * The connecting weights between #input and #output. This is two-dimensional, the first dimension
 * is one per input, plus a *bias* (the last item in each "row"); the second dimension is set by
 * number of outputs.
 * @return [NArray<sfloat>] two-dimensional array of [#num_inputs+1, #num_outputs] single-precision floats
 */
VALUE mlp_layer_object_weights( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_weights;
}

/* @!attribute [r] input_layer
 * The current input layer.
 * @return [RuNeNe::Layer::FeedForward,nil] a nil value means this is the first layer in a connected set.
 */
VALUE mlp_layer_object_input_layer( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->input_layer;
}

/* @!attribute [r] output_layer
 * The current output layer.
 * @return [RuNeNe::Layer::FeedForward,nil] a nil value means this is the last layer in a connected set.
 */
VALUE mlp_layer_object_output_layer( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->output_layer;
}

/* @!attribute [r] output_deltas
 * Array of differences calculated during training.
 * @return [NArray<sfloat>] one-dimensional array of #num_outputs single-precision floats
 */
VALUE mlp_layer_object_output_deltas( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_output_deltas;
}

/* @!attribute [r] weights_last_deltas
 * The last corrections made to each weight. The values are used with training that uses momentum.
 * @return [NArray<sfloat>] two-dimensional array of [#num_inputs+1, #num_outputs] single-precision floats
 */
VALUE mlp_layer_object_weights_last_deltas( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  return mlp_layer->narr_weights_last_deltas;
}

/* @overload init_weights( *limits )
 * Sets the weights array to new random values. The limits are optional floats that set
 * the range. Default range (no params) is *-0.8..0.8*. With one param *x*, the range is *-x..x*.
 * With two params *x* ,*y*, the range is *x..y*.
 * @param [Float] limits supply 0, 1 or 2 Float values
 * @return [nil]
 */
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

/* @overload set_input( input_array )
 * Sets the input to the layer. Any existing inputs or input layers are dicsonnected.
 * @param [NArray] input_array one-dimensional array of #num_inputs numbers
 * @return [NArray<sfloat>] the new input array (may be same as parameter)
 */
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

/* @overload attach_input_layer( input_layer )
 * Sets the input layer to this layer. Any existing inputs or input layers are disconnected.
 * The input layer also has this layer set as its output_layer.
 * @param [RuNeNe::Layer::FeedForward] input_layer must have #num_outputs equal to #num_inputs of this layer
 * @return [RuNeNe::Layer::FeedForward] the new input layer (always the same as parameter)
 */
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

/* @overload attach_input_layer( output_layer )
 * Sets the output layer to this layer. Any existing output layer is disconnected.
 * The output layer also has this layer set as its input_layer.
 * @param [RuNeNe::Layer::FeedForward] output_layer must have #num_inputs equal to #num_outputs of this layer
 * @return [RuNeNe::Layer::FeedForward] the new output layer (always the same as parameter)
 */
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

/* @overload run( )
 * Sets values in #output based on current values in #input, using the #weights array
 * and #transfer.
 * @return [NArray<sfloat>] same as #output
 */
VALUE mlp_layer_object_run( VALUE self ) {
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );

  if ( NIL_P( mlp_layer->narr_input ) ) {
    rb_raise( rb_eArgError, "No input. Cannot run MLP layer." );
  }

  p_mlp_layer_run( mlp_layer );

  return mlp_layer->narr_output;
}

/* @overload ms_error( target )
 * Calculates the mean squared error of the output compared to the target array.
 * @param [NArray] target one-dimensional array of #num_outputs single-precision floats
 * @return [Float]
 */
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

/* @overload calc_output_deltas( target )
 * Sets values in #output_deltas array based on current values in #output compared to target
 * array. Calculating these values is one step in the backpropagation algorithm.
 * @param [NArray] target one-dimensional array of #num_outputs single-precision floats
 * @return [NArray<sfloat>] the #output_deltas
 */
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

/* @overload backprop_deltas( )
 * Sets values in #output_deltas array of the #input_layer, based on current values
 * in #output_deltas in this layer and the #weights and #input. Calculating these values
 * is one step in the backpropagation algorithm.
 * @return [NArray<sfloat>] the #output_deltas from the #input_layer
 */
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

/* @overload update_weights( learning_rate, momentum = 0.0 )
 * Alters values in #weights based on #output_deltas. The amount of change is also stored
 * in #weights_last_deltas (which is also returned)
 * @param [Float] learning_rate multiplier for amount of adjustment, 0.0..1000.0
 * @param [Float] momentum amount of previous weight change to add in 0.0..0.95
 * @return [NArray<sfloat>] value of #weights_last_deltas after calculation
 */
VALUE mlp_layer_object_update_weights( int argc, VALUE* argv, VALUE self ) {
  VALUE learning_rate, momentum;
  MLP_Layer *mlp_layer = get_mlp_layer_struct( self );
  float eta, m;
  rb_scan_args( argc, argv, "11", &learning_rate, &momentum );

  eta = NUM2FLT( learning_rate );
  if ( eta < 0.0 || eta > 1000.0 ) {
    rb_raise( rb_eArgError, "Learning rate %0.6f out of bounds (0.0 to 1000.0).", eta );
  }

  m = 0.0;

  if ( ! NIL_P(momentum) ) {
    m = NUM2FLT( momentum );
    if ( m < 0.0 || m > 0.95 ) {
      rb_raise( rb_eArgError, "Momentum %0.6f out of bounds (0.0 to 0.95).", m );
    }
  }

  p_mlp_layer_update_weights( mlp_layer, eta, m );

  return mlp_layer->narr_weights_last_deltas;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mlp_layer_class() {
  // FeedForward instantiation and class methods
  rb_define_alloc_func( RuNeNe_Layer_FeedForward, mlp_layer_alloc );
  rb_define_method( RuNeNe_Layer_FeedForward, "initialize", mlp_layer_class_initialize, -1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "initialize_copy", mlp_layer_class_initialize_copy, 1 );
  rb_define_singleton_method( RuNeNe_Layer_FeedForward, "from_weights", mlp_layer_class_from_weights, -1 );

  // FeedForward attributes
  rb_define_method( RuNeNe_Layer_FeedForward, "num_inputs", mlp_layer_object_num_inputs, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "num_outputs", mlp_layer_object_num_outputs, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "transfer", mlp_layer_object_transfer, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "input", mlp_layer_object_input, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "output", mlp_layer_object_output, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "weights", mlp_layer_object_weights, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "input_layer", mlp_layer_object_input_layer, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "output_layer", mlp_layer_object_output_layer, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "output_deltas", mlp_layer_object_output_deltas, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "weights_last_deltas", mlp_layer_object_weights_last_deltas, 0 );

  // FeedForward methods
  rb_define_method( RuNeNe_Layer_FeedForward, "init_weights", mlp_layer_object_init_weights, -1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "set_input", mlp_layer_object_set_input, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "attach_input_layer", mlp_layer_object_attach_input_layer, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "attach_output_layer", mlp_layer_object_attach_output_layer, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "run", mlp_layer_object_run, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "ms_error", mlp_layer_object_ms_error, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "calc_output_deltas", mlp_layer_object_calc_output_deltas, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "backprop_deltas", mlp_layer_object_backprop_deltas, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "update_weights", mlp_layer_object_update_weights, -1 );
}
