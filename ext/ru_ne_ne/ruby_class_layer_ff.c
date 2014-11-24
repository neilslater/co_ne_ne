// ext/ru_ne_ne/ruby_class_layer_ff.c

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for multi-layer perceptron code - the deeper implementation is in
//  layer_ff_raw.c and network_raw.c
//

#include "ruby_class_layer_ff.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

inline VALUE layer_ff_as_ruby_class( Layer_FF *layer_ff, VALUE klass ) {
  return Data_Wrap_Struct( klass, layer_ff__gc_mark, layer_ff__destroy, layer_ff );
}

VALUE layer_ff_alloc(VALUE klass) {
  return layer_ff_as_ruby_class( layer_ff__create(), klass );
}

inline Layer_FF *get_layer_ff_struct( VALUE obj ) {
  Layer_FF *layer_ff;
  Data_Get_Struct( obj, Layer_FF, layer_ff );
  return layer_ff;
}

VALUE layer_ff_new_ruby_object( int n_inputs, int n_outputs, transfer_type tfn ) {
  Layer_FF *layer_ff;
  VALUE rv_layer_ff = layer_ff_alloc( RuNeNe_Layer_FeedForward );
  layer_ff = get_layer_ff_struct( rv_layer_ff );

  layer_ff->num_inputs = n_inputs;
  layer_ff->num_outputs = n_outputs;
  layer_ff->transfer_fn = tfn;

  layer_ff__new_narrays( layer_ff );
  // TODO: Needs to vary according to layer size
  layer_ff__init_weights( layer_ff, -0.8, 0.8 );

  return rv_layer_ff;
}

VALUE layer_ff_clone_ruby_object( VALUE orig ) {
  volatile VALUE copy;
  Layer_FF *layer_ff_copy;
  Layer_FF *layer_ff_orig;
  layer_ff_orig = get_layer_ff_struct( orig );

  copy =  layer_ff_alloc( RuNeNe_Layer_FeedForward );
  layer_ff_copy = get_layer_ff_struct( copy );

  layer_ff_copy->num_inputs = layer_ff_orig->num_inputs;
  layer_ff_copy->num_outputs = layer_ff_orig->num_outputs;
  layer_ff_copy->transfer_fn = layer_ff_orig->transfer_fn;

  layer_ff_copy->narr_input = Qnil;
  layer_ff_copy->input_layer = Qnil;
  layer_ff_copy->output_layer = Qnil;

  layer_ff_copy->narr_output = na_clone( layer_ff_orig->narr_output );
  layer_ff_copy->narr_weights = na_clone( layer_ff_orig->narr_weights );
  layer_ff_copy->narr_output_deltas = na_clone( layer_ff_orig->narr_output_deltas );
  layer_ff_copy->narr_weights_last_deltas = na_clone( layer_ff_orig->narr_weights_last_deltas );
  layer_ff_copy->narr_output_slope = na_clone( layer_ff_orig->narr_output_slope );

  return copy;
}

void assert_value_wraps_layer_ff( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)layer_ff__destroy) {
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

void set_transfer_fn_from_symbol( Layer_FF *layer_ff , VALUE tfn_type ) {
  layer_ff->transfer_fn = transfer_fn_from_symbol( tfn_type );
}

void assert_not_in_output_chain( Layer_FF *layer_ff, VALUE unexpected_layer ) {
  Layer_FF *next_layer_ff = layer_ff;
  while ( ! NIL_P( next_layer_ff->output_layer ) ) {
    if ( next_layer_ff->output_layer == unexpected_layer ) {
      rb_raise( rb_eArgError, "Attempt to create a circular network." );
    }
    next_layer_ff = get_layer_ff_struct( next_layer_ff->output_layer );
  }
  return;
}

void assert_not_in_input_chain( Layer_FF *layer_ff, VALUE unexpected_layer ) {
  Layer_FF *next_layer_ff = layer_ff;
  while ( ! NIL_P( next_layer_ff->input_layer ) ) {
    if ( next_layer_ff->input_layer == unexpected_layer ) {
      rb_raise( rb_eArgError, "Attempt to create a circular network." );
    }
    next_layer_ff = get_layer_ff_struct( next_layer_ff->input_layer );
  }
  return;
}

VALUE layer_ff_new_ruby_object_from_weights( VALUE weights, transfer_type tfn ) {
  Layer_FF *layer_ff;
  struct NARRAY *na_weights;
  VALUE rv_layer_ff = layer_ff_alloc( RuNeNe_Layer_FeedForward );
  layer_ff = get_layer_ff_struct( rv_layer_ff );

  GetNArray( weights, na_weights );
  layer_ff->num_inputs = na_weights->shape[0] - 1;
  layer_ff->num_outputs = na_weights->shape[1];
  layer_ff->transfer_fn = tfn;
  layer_ff__init_from_weights( layer_ff, weights );

  return rv_layer_ff;
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
VALUE layer_ff_class_initialize( int argc, VALUE* argv, VALUE self ) {
  VALUE n_ins, n_outs, tfn_type;
  Layer_FF *layer_ff = get_layer_ff_struct( self );
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

  layer_ff->num_inputs = i;
  layer_ff->num_outputs = o;

  set_transfer_fn_from_symbol( layer_ff, tfn_type );

  layer_ff__new_narrays( layer_ff );
  layer_ff__init_weights( layer_ff, -0.8, 0.8 );

  return self;
}

/* @overload clone
 * When cloned, the returned Layer has deep copies of weights and outputs,
 * and is *not* connected to the inputs and outputs that the original was.
 * @return [RuNeNe::Layer::FeedForward] new layer same weights and transfer function.
 */
VALUE layer_ff_class_initialize_copy( VALUE copy, VALUE orig ) {
  Layer_FF *layer_ff_copy;
  Layer_FF *layer_ff_orig;

  if (copy == orig) return copy;
  layer_ff_copy = get_layer_ff_struct( copy );
  layer_ff_orig = get_layer_ff_struct( orig );

  layer_ff_copy->num_inputs = layer_ff_orig->num_inputs;
  layer_ff_copy->num_outputs = layer_ff_orig->num_outputs;
  layer_ff_copy->transfer_fn = layer_ff_orig->transfer_fn;

  layer_ff_copy->narr_input = Qnil;
  layer_ff_copy->input_layer = Qnil;
  layer_ff_copy->output_layer = Qnil;

  layer_ff_copy->narr_output = na_clone( layer_ff_orig->narr_output );
  layer_ff_copy->narr_weights = na_clone( layer_ff_orig->narr_weights );
  layer_ff_copy->narr_output_deltas = na_clone( layer_ff_orig->narr_output_deltas );
  layer_ff_copy->narr_weights_last_deltas = na_clone( layer_ff_orig->narr_weights_last_deltas );
  layer_ff_copy->narr_output_slope = na_clone( layer_ff_orig->narr_output_slope );

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
VALUE layer_ff_class_from_weights( int argc, VALUE* argv, VALUE self ) {
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

  return layer_ff_new_ruby_object_from_weights( val_weights, transfer_fn_from_symbol( tfn_type ) );
}

/* @!attribute [r] num_inputs
 * Number of inputs to the layer. This affects the size of arrays when setting the input.
 * @return [Integer]
 */
VALUE layer_ff_object_num_inputs( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  return INT2FIX( layer_ff->num_inputs );
}

/* @!attribute [r] num_outputs
 * Number of outputs from the layer. This affects the size of arrays for training targets.
 * @return [Integer]
 */
VALUE layer_ff_object_num_outputs( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  return INT2FIX( layer_ff->num_outputs );
}

/* @!attribute [r] transfer
 * The RuNeNe::Transfer *Module* that is used for transfer methods when the layer is #run.
 * @return [Module]
 */
VALUE layer_ff_object_transfer( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  VALUE t;
  switch ( layer_ff->transfer_fn ) {
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
VALUE layer_ff_object_input( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  return layer_ff->narr_input;
}

/* @!attribute [r] output
 * The current output array.
 * @return [NArray<sfloat>] one-dimensional array of #num_outputs single-precision floats
 */
VALUE layer_ff_object_output( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  return layer_ff->narr_output;
}

/* @!attribute [r] weights
 * The connecting weights between #input and #output. This is two-dimensional, the first dimension
 * is one per input, plus a *bias* (the last item in each "row"); the second dimension is set by
 * number of outputs.
 * @return [NArray<sfloat>] two-dimensional array of [#num_inputs+1, #num_outputs] single-precision floats
 */
VALUE layer_ff_object_weights( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  return layer_ff->narr_weights;
}

/* @!attribute [r] input_layer
 * The current input layer.
 * @return [RuNeNe::Layer::FeedForward,nil] a nil value means this is the first layer in a connected set.
 */
VALUE layer_ff_object_input_layer( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  return layer_ff->input_layer;
}

/* @!attribute [r] output_layer
 * The current output layer.
 * @return [RuNeNe::Layer::FeedForward,nil] a nil value means this is the last layer in a connected set.
 */
VALUE layer_ff_object_output_layer( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  return layer_ff->output_layer;
}

/* @!attribute [r] output_deltas
 * Array of differences calculated during training.
 * @return [NArray<sfloat>] one-dimensional array of #num_outputs single-precision floats
 */
VALUE layer_ff_object_output_deltas( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  return layer_ff->narr_output_deltas;
}

/* @!attribute [r] weights_last_deltas
 * The last corrections made to each weight. The values are used with training that uses momentum.
 * @return [NArray<sfloat>] two-dimensional array of [#num_inputs+1, #num_outputs] single-precision floats
 */
VALUE layer_ff_object_weights_last_deltas( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  return layer_ff->narr_weights_last_deltas;
}

/* @overload init_weights( *limits )
 * Sets the weights array to new random values. The limits are optional floats that set
 * the range. Default range (no params) is *-0.8..0.8*. With one param *x*, the range is *-x..x*.
 * With two params *x* ,*y*, the range is *x..y*.
 * @param [Float] limits supply 0, 1 or 2 Float values
 * @return [nil]
 */
VALUE layer_ff_object_init_weights( int argc, VALUE* argv, VALUE self ) {
  VALUE minw, maxw;
  Layer_FF *layer_ff = get_layer_ff_struct( self );
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

  layer_ff__init_weights( layer_ff, min_weight, max_weight );

  return Qnil;
}

/* @overload set_input( input_array )
 * Sets the input to the layer. Any existing inputs or input layers are dicsonnected.
 * @param [NArray] input_array one-dimensional array of #num_inputs numbers
 * @return [NArray<sfloat>] the new input array (may be same as parameter)
 */
VALUE layer_ff_object_set_input( VALUE self, VALUE new_input ) {
  struct NARRAY *na_input;
  volatile VALUE val_input;
  Layer_FF *layer_ff = get_layer_ff_struct( self );

  val_input = na_cast_object(new_input, NA_SFLOAT);
  GetNArray( val_input, na_input );

  if ( na_input->rank != 1 ) {
    rb_raise( rb_eArgError, "Inputs rank should be 1, but got %d", na_input->rank );
  }

  if ( na_input->total != layer_ff->num_inputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer input size %d", na_input->total, layer_ff->num_inputs );
  }

  layer_ff__set_input( layer_ff, val_input );

  return val_input;
}

/* @overload attach_input_layer( input_layer )
 * Sets the input layer to this layer. Any existing inputs or input layers are disconnected.
 * The input layer also has this layer set as its output_layer.
 * @param [RuNeNe::Layer::FeedForward] input_layer must have #num_outputs equal to #num_inputs of this layer
 * @return [RuNeNe::Layer::FeedForward] the new input layer (always the same as parameter)
 */
VALUE layer_ff_object_attach_input_layer( VALUE self, VALUE new_input_layer ) {
  Layer_FF *s_new_input_layer_ff;
  Layer_FF *s_old_output_layer_ff, *s_old_input_layer_ff;
  Layer_FF *layer_ff = get_layer_ff_struct( self );

  if ( layer_ff->locked_input > 0 ) {
    rb_raise( rb_eArgError, "Layer has been marked as 'first layer' and may not have another input layer attached." );
  }

  assert_value_wraps_layer_ff( new_input_layer );
  s_new_input_layer_ff = get_layer_ff_struct( new_input_layer );

  if ( s_new_input_layer_ff->num_outputs != layer_ff->num_inputs ) {
    rb_raise( rb_eArgError, "Input layer output size %d does not match layer input size %d", s_new_input_layer_ff->num_outputs, layer_ff->num_inputs );
  }

  assert_not_in_input_chain( s_new_input_layer_ff, self );

  if ( ! NIL_P( layer_ff->input_layer ) ) {
    // This layer has an existing input layer, it needs to stop pointing its output here
    s_old_input_layer_ff = get_layer_ff_struct( layer_ff->input_layer );
    s_old_input_layer_ff->output_layer = Qnil;
  }

  layer_ff->narr_input = s_new_input_layer_ff->narr_output;
  layer_ff->input_layer = new_input_layer;

  if ( ! NIL_P( s_new_input_layer_ff->output_layer ) ) {
    // The new input layer was previously attached elsewhere. This needs to be disconnected too
    s_old_output_layer_ff = get_layer_ff_struct( s_new_input_layer_ff->output_layer );
    s_old_output_layer_ff->narr_input = Qnil;
    s_old_output_layer_ff->input_layer = Qnil;
  }
  s_new_input_layer_ff->output_layer = self;

  return new_input_layer;
}

/* @overload attach_input_layer( output_layer )
 * Sets the output layer to this layer. Any existing output layer is disconnected.
 * The output layer also has this layer set as its input_layer.
 * @param [RuNeNe::Layer::FeedForward] output_layer must have #num_inputs equal to #num_outputs of this layer
 * @return [RuNeNe::Layer::FeedForward] the new output layer (always the same as parameter)
 */
VALUE layer_ff_object_attach_output_layer( VALUE self, VALUE new_output_layer ) {
  Layer_FF *s_new_output_layer_ff;
  Layer_FF *s_old_output_layer_ff, *s_old_input_layer_ff;
  Layer_FF *layer_ff = get_layer_ff_struct( self );

  assert_value_wraps_layer_ff( new_output_layer );
  s_new_output_layer_ff = get_layer_ff_struct( new_output_layer );

  if ( s_new_output_layer_ff->locked_input > 0 ) {
    rb_raise( rb_eArgError, "Target layer has been marked as 'first layer' and may not have another input layer attached." );
  }

  if ( s_new_output_layer_ff->num_inputs != layer_ff->num_outputs ) {
    rb_raise( rb_eArgError, "Output layer input size %d does not match layer output size %d", s_new_output_layer_ff->num_inputs, layer_ff->num_outputs );
  }

  assert_not_in_output_chain( s_new_output_layer_ff, self );

  if ( ! NIL_P( layer_ff->output_layer ) ) {
    // This layer has an existing output layer, it needs to stop pointing its input here
    s_old_output_layer_ff = get_layer_ff_struct( layer_ff->output_layer );
    s_old_output_layer_ff->input_layer = Qnil;
    s_old_output_layer_ff->narr_input = Qnil;
  }

  layer_ff->output_layer = new_output_layer;

  if ( ! NIL_P( s_new_output_layer_ff->input_layer ) ) {
    // The new output layer was previously attached elsewhere. This needs to be disconnected too
    s_old_input_layer_ff = get_layer_ff_struct( s_new_output_layer_ff->input_layer );
    s_old_input_layer_ff->output_layer = Qnil;
  }
  s_new_output_layer_ff->input_layer = self;
  s_new_output_layer_ff->narr_input = layer_ff->narr_output;

  return new_output_layer;
}

/* @overload run( )
 * Sets values in #output based on current values in #input, using the #weights array
 * and #transfer.
 * @return [NArray<sfloat>] same as #output
 */
VALUE layer_ff_object_run( VALUE self ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );

  if ( NIL_P( layer_ff->narr_input ) ) {
    rb_raise( rb_eArgError, "No input. Cannot run MLP layer." );
  }

  layer_ff__run( layer_ff );

  return layer_ff->narr_output;
}

/* @overload ms_error( target )
 * Calculates the mean squared error of the output compared to the target array.
 * @param [NArray] target one-dimensional array of #num_outputs single-precision floats
 * @return [Float]
 */
VALUE layer_ff_object_ms_error( VALUE self, VALUE target ) {
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  volatile VALUE val_target;
  Layer_FF *layer_ff = get_layer_ff_struct( self );

  val_target = na_cast_object(target, NA_SFLOAT);
  GetNArray( val_target, na_target );

  if ( na_target->rank != 1 ) {
    rb_raise( rb_eArgError, "Target output rank should be 1, but got %d", na_target->rank );
  }

  if ( na_target->total != layer_ff->num_outputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer output size %d", na_target->total, layer_ff->num_outputs );
  }

  GetNArray( layer_ff->narr_output, na_output );

  return FLT2NUM( mean_square_error( layer_ff->num_outputs, (float *) na_output->ptr,  (float *) na_target->ptr ) );
}

/* @overload calc_output_deltas( target )
 * Sets values in #output_deltas array based on current values in #output compared to target
 * array. Calculating these values is one step in the backpropagation algorithm.
 * @param [NArray] target one-dimensional array of #num_outputs single-precision floats
 * @return [NArray<sfloat>] the #output_deltas
 */
VALUE layer_ff_object_calc_output_deltas( VALUE self, VALUE target ) {
  struct NARRAY *na_target;
  volatile VALUE val_target;
  Layer_FF *layer_ff = get_layer_ff_struct( self );

  val_target = na_cast_object(target, NA_SFLOAT);
  GetNArray( val_target, na_target );

  if ( na_target->rank != 1 ) {
    rb_raise( rb_eArgError, "Target output rank should be 1, but got %d", na_target->rank );
  }

  if ( na_target->total != layer_ff->num_outputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer output size %d", na_target->total, layer_ff->num_outputs );
  }

  layer_ff__calc_output_deltas( layer_ff, val_target );

  return layer_ff->narr_output_deltas;
}

/* @overload backprop_deltas( )
 * Sets values in #output_deltas array of the #input_layer, based on current values
 * in #output_deltas in this layer and the #weights and #input. Calculating these values
 * is one step in the backpropagation algorithm.
 * @return [NArray<sfloat>] the #output_deltas from the #input_layer
 */
VALUE layer_ff_object_backprop_deltas( VALUE self ) {
  Layer_FF *layer_ff_input;
  Layer_FF *layer_ff = get_layer_ff_struct( self );

  if ( NIL_P( layer_ff->input_layer ) ) {
    rb_raise( rb_eArgError, "No input layer. Cannot run MLP backpropagation." );
  }

  layer_ff_input = get_layer_ff_struct( layer_ff->input_layer );

  layer_ff__backprop_deltas( layer_ff, layer_ff_input );

  return layer_ff_input->narr_output_deltas;
}

/* @overload update_weights( learning_rate, momentum = 0.0 )
 * Alters values in #weights based on #output_deltas. The amount of change is also stored
 * in #weights_last_deltas (which is also returned)
 * @param [Float] learning_rate multiplier for amount of adjustment, 0.0..1000.0
 * @param [Float] momentum amount of previous weight change to add in 0.0..0.95
 * @return [NArray<sfloat>] value of #weights_last_deltas after calculation
 */
VALUE layer_ff_object_update_weights( int argc, VALUE* argv, VALUE self ) {
  VALUE learning_rate, momentum;
  Layer_FF *layer_ff = get_layer_ff_struct( self );
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

  layer_ff__update_weights( layer_ff, eta, m );

  return layer_ff->narr_weights_last_deltas;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void init_layer_ff_class() {
  // FeedForward instantiation and class methods
  rb_define_alloc_func( RuNeNe_Layer_FeedForward, layer_ff_alloc );
  rb_define_method( RuNeNe_Layer_FeedForward, "initialize", layer_ff_class_initialize, -1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "initialize_copy", layer_ff_class_initialize_copy, 1 );
  rb_define_singleton_method( RuNeNe_Layer_FeedForward, "from_weights", layer_ff_class_from_weights, -1 );

  // FeedForward attributes
  rb_define_method( RuNeNe_Layer_FeedForward, "num_inputs", layer_ff_object_num_inputs, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "num_outputs", layer_ff_object_num_outputs, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "transfer", layer_ff_object_transfer, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "input", layer_ff_object_input, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "output", layer_ff_object_output, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "weights", layer_ff_object_weights, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "input_layer", layer_ff_object_input_layer, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "output_layer", layer_ff_object_output_layer, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "output_deltas", layer_ff_object_output_deltas, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "weights_last_deltas", layer_ff_object_weights_last_deltas, 0 );

  // FeedForward methods
  rb_define_method( RuNeNe_Layer_FeedForward, "init_weights", layer_ff_object_init_weights, -1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "set_input", layer_ff_object_set_input, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "attach_input_layer", layer_ff_object_attach_input_layer, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "attach_output_layer", layer_ff_object_attach_output_layer, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "run", layer_ff_object_run, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "ms_error", layer_ff_object_ms_error, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "calc_output_deltas", layer_ff_object_calc_output_deltas, 1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "backprop_deltas", layer_ff_object_backprop_deltas, 0 );
  rb_define_method( RuNeNe_Layer_FeedForward, "update_weights", layer_ff_object_update_weights, -1 );
}
