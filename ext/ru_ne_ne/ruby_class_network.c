// ext/ru_ne_ne/ruby_class_network.c

#include "ruby_class_network.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for multi-layer perceptron code - the deeper implementation is in
//  struct_layer_ff.c and struct_network.c
//

inline VALUE network_as_ruby_class( MLP_Network *network , VALUE klass ) {
  return Data_Wrap_Struct( klass, p_network_gc_mark, p_network_destroy, network );
}

VALUE network_alloc(VALUE klass) {
  return network_as_ruby_class( p_network_create(), klass );
}

inline MLP_Network *get_network_struct( VALUE obj ) {
  MLP_Network *network;
  Data_Get_Struct( obj, MLP_Network, network );
  return network;
}

void assert_value_wraps_network( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)p_network_destroy) {
    rb_raise( rb_eTypeError, "Expected a Network object, but got something else" );
  }
}

VALUE network_new_ruby_object_from_layer( VALUE layer, float eta, float momentum ) {
  MLP_Network *network;
  s_Layer_FF *layer_ff;
  volatile VALUE network_ruby = network_alloc( RuNeNe_Network );
  network = get_network_struct( network_ruby );
  Data_Get_Struct( layer, s_Layer_FF, layer_ff );

  p_layer_ff_clear_input( layer_ff );
  layer_ff->locked_input = 1;
  network->first_layer = layer;
  network->eta = eta;
  network->momentum = momentum;

  return network_ruby;
}


/* Document-class:  RuNeNe::Network
 *
 * An object of this class represents a feed-forward network consisting of one or more layers. It
 * can be trained by repeatedly showing it example inputs with target outputs. The training
 * process alters weights in each layer.
 *
 * The first layer of the network has special status. To ensure it remains the first layer, it
 * may not have another layer attached as its input. However, other changes to layers in the
 * network are generally allowed. This includes attaching new or alternative layers. Properties
 * of the network as a whole are calculated dynamically from the attached layers.
 *
 * RuNeNe::Network supports persisting objects via Marshal.
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Network method definitions
//

/* @overload initialize( num_inputs, hidden_layers, num_outputs )
 * Creates a new network and initializes the weights in all layers.
 * @param [Integer] num_inputs size of input array for first layer
 * @param [Array<Integer>] hidden_layers sizes of output arrays for each hidden layer
 * @param [Integer] num_outputs size of output array for last layer
 * @return [RuNeNe::Network] new network consisting of new layers, with random weights
 */
VALUE network_class_initialize( VALUE self, VALUE num_inputs, VALUE hidden_layers, VALUE num_outputs ) {
  int ninputs, noutputs, i, nhlayers, hlsize, *layer_sizes;
  MLP_Network *network = get_network_struct( self );
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

  p_network_init_layers( network, nhlayers + 1, layer_sizes );

  xfree( layer_sizes );
  return self;
}

/* @overload clone
 * When cloned, the new Network has deep copies of all layers (which in
 * turn have deep copies of all weights etc)
 * @return [RuNeNe::Network] new network with same weights.
 */
VALUE network_class_initialize_copy( VALUE copy, VALUE orig ) {
  MLP_Network *network_copy;
  MLP_Network *network_orig;
  volatile VALUE orig_layer;
  volatile VALUE copy_layer;
  volatile VALUE copy_layer_prev;
  s_Layer_FF *layer_ff_orig;
  s_Layer_FF *layer_ff_copy;
  s_Layer_FF *layer_ff_copy_prev;

  if (copy == orig) return copy;
  network_copy = get_network_struct( copy );
  network_orig = get_network_struct( orig );
  network_copy->eta = network_orig->eta;
  network_copy->momentum = network_orig->momentum;

  // Copy first layer
  orig_layer = network_orig->first_layer;
  copy_layer = layer_ff_clone_ruby_object( orig_layer );
  network_copy->first_layer = copy_layer;
  Data_Get_Struct( orig_layer, s_Layer_FF, layer_ff_orig );
  Data_Get_Struct( copy_layer, s_Layer_FF, layer_ff_copy_prev );
  copy_layer_prev = copy_layer;

  // Copy and attach each layer in turn
  while ( ! NIL_P(layer_ff_orig->output_layer) ) {
    orig_layer = layer_ff_orig->output_layer;
    copy_layer = layer_ff_clone_ruby_object( orig_layer );
    Data_Get_Struct( orig_layer, s_Layer_FF, layer_ff_orig );
    Data_Get_Struct( copy_layer, s_Layer_FF, layer_ff_copy );

    layer_ff_copy_prev->output_layer = copy_layer;
    layer_ff_copy->input_layer = copy_layer_prev;
    layer_ff_copy->narr_input = layer_ff_copy_prev->narr_output;

    copy_layer_prev = copy_layer;
    layer_ff_copy_prev = layer_ff_copy;
  }

  return copy;
}

/* @overload from_layer( layer )
 * Creates a new network with supplied layer as the first layer. The layer can already
 * be connected, or you can add new layers later using RuNeNe::Layer::FeedForward#attach_output_layer
 * @param [RuNeNe::Layer::FeedForward] layer first layer of new network
 * @return [RuNeNe::Network] new network
 */
VALUE network_class_from_layer( VALUE self, VALUE layer ) {
  s_Layer_FF *layer_ff;
  assert_value_wraps_layer_ff( layer );
  Data_Get_Struct( layer, s_Layer_FF, layer_ff );
  if ( ! NIL_P( layer_ff->input_layer ) ) {
    rb_raise( rb_eArgError, "Cannot create network from layer with an attached input layer." );
  }
  return network_new_ruby_object_from_layer( layer, 1.0, 0.5 );
}

/* @overload num_layers
 * @!attribute [r] num_layers
 * Total number of layers in the network.
 * @return [Integer]
 */
VALUE network_object_num_layers( VALUE self ) {
  MLP_Network *network = get_network_struct( self );
  return INT2NUM( p_network_count_layers( network ) );
}

/* @overload layers
 * @!attribute [r] layers
 * Array of layer objects within the network, in order input to output.
 * @return [Array<RuNeNe::Layer::FeedForward]
 */
VALUE network_object_layers( VALUE self ) {
  int num_layers, count;
  VALUE layer_object, all_layers;
  s_Layer_FF *layer_ff;
  MLP_Network *network = get_network_struct( self );

  num_layers = p_network_count_layers( network );

  all_layers = rb_ary_new2( num_layers );
  count = 0;
  layer_object = network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    rb_ary_store( all_layers, count, layer_object );
    count++;
    Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );
    layer_object = layer_ff->output_layer;
  }

  return all_layers;
}

/* @overload init_weights( *limits )
 * Sets weights arrays in all layers to new random values. The limits are optional floats that set
 * the range. Default range (no params) is *-0.8..0.8*. With one param *x*, the range is *-x..x*.
 * With two params *x* ,*y*, the range is *x..y*.
 * @param [Float] limits supply 0, 1 or 2 Float values
 * @return [nil]
 */
VALUE network_object_init_weights( int argc, VALUE* argv, VALUE self ) {
  VALUE minw, maxw;
  float min_weight, max_weight;
  MLP_Network *network = get_network_struct( self );

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

  p_network_init_layer_weights( network, min_weight, max_weight );

  return Qnil;
}

/* @overload num_outputs
 * @!attribute [r] num_outputs
 * Size of output array of last layer in network. This is the size of array that should be
 * used for training targets.
 * @return [Integer]
 */
VALUE network_object_num_outputs( VALUE self ) {
  MLP_Network *network = get_network_struct( self );
  return INT2NUM( p_network_num_outputs( network ) );
}

/* @overload num_inputs
 * @!attribute [r] num_inputs
 * Size of input array of first layer in network. This is the size of array that should be
 * used for all inputs to the network, and cannot be altered after the network is created.
 * @return [Integer]
 */
VALUE network_object_num_inputs( VALUE self ) {
  MLP_Network *network = get_network_struct( self );
  return INT2NUM( p_network_num_inputs( network ) );
}

/* @overload output
 * @!attribute [r] output
 * Current output from the network.
 * @return [NArray<sfloat>] one-dimensional array of single-precision floats
 */
VALUE network_object_output( VALUE self ) {
  s_Layer_FF *layer_ff;
  MLP_Network *network = get_network_struct( self );

  layer_ff = p_network_last_layer_ff( network );
  return layer_ff->narr_output;
}

/* @overload input
 * @!attribute [r] input
 * Current input to the network.
 * @return [NArray<sfloat>,nil] one-dimensional array of single-precision floats
 */
VALUE network_object_input( VALUE self ) {
  s_Layer_FF *layer_ff;
  MLP_Network *network = get_network_struct( self );

  Data_Get_Struct( network->first_layer, s_Layer_FF, layer_ff );
  return layer_ff->narr_input;
}

/* @overload run( new_input )
 * Uses supplied parameter as new input to the first layer and runs each layer in
 * turn until the output array is set.
 * @param [NArray] new_input one-dimensional array of #num_inputs single-precision floats
 * @return [NArray<sfloat>] the #output array
 */
VALUE network_object_run( VALUE self, VALUE new_input ) {
  struct NARRAY *na_input;
  volatile VALUE val_input;
  volatile VALUE layer_object;
  s_Layer_FF *layer_ff;
  MLP_Network *network = get_network_struct( self );

  layer_object = network->first_layer;
  Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );

  val_input = na_cast_object(new_input, NA_SFLOAT);
  GetNArray( val_input, na_input );

  if ( na_input->rank != 1 ) {
    rb_raise( rb_eArgError, "Inputs rank should be 1, but got %d", na_input->rank );
  }

  if ( na_input->total != layer_ff->num_inputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer input size %d", na_input->total, layer_ff->num_inputs );
  }

  p_layer_ff_set_input( layer_ff, val_input );
  p_network_run( network );
  layer_ff = p_network_last_layer_ff( network );
  return layer_ff->narr_output;
}

/* @overload ms_error( target )
 * Calculates the mean squared error of the network's output compared to the target array.
 * @param [NArray] target one-dimensional array of #num_outputs single-precision floats
 * @return [Float]
 */
VALUE network_object_ms_error( VALUE self, VALUE target ) {
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  volatile VALUE val_target;
  s_Layer_FF *layer_ff;
  MLP_Network *network = get_network_struct( self );

  val_target = na_cast_object(target, NA_SFLOAT);
  GetNArray( val_target, na_target );

  if ( na_target->rank != 1 ) {
    rb_raise( rb_eArgError, "Target rank should be 1, but got %d", na_target->rank );
  }

  layer_ff = p_network_last_layer_ff( network );

  if ( na_target->total != layer_ff->num_outputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match network output size %d", na_target->total, layer_ff->num_outputs );
  }

  GetNArray( layer_ff->narr_output, na_output );

  return FLT2NUM( core_mean_square_error( layer_ff->num_outputs, (float *) na_output->ptr,  (float *) na_target->ptr ) );
}

/* @overload train_once( new_input, target )
 * Takes a pair of input and desired output, runs the network forward, calculates and backpropagates
 * the error, then updates the weights in each layer. A single "unit of learning".
 * @param [NArray] new_input one-dimensional array of #num_inputs single-precision floats
 * @param [NArray] target one-dimensional array of #num_outputs single-precision floats
 * @return [nil]
 */
VALUE network_object_train_once( VALUE self, VALUE new_input, VALUE target ) {
  struct NARRAY *na_input;
  volatile VALUE val_input;
  struct NARRAY *na_target;

  volatile VALUE val_target;
  volatile VALUE layer_object;

  s_Layer_FF *layer_ff;
  MLP_Network *network = get_network_struct( self );

  ////////////////////////////////////////////////////////////////////////////////////
  // Check input is valid
  layer_object = network->first_layer;
  Data_Get_Struct( layer_object, s_Layer_FF, layer_ff );

  val_input = na_cast_object(new_input, NA_SFLOAT);
  GetNArray( val_input, na_input );

  if ( na_input->rank != 1 ) {
    rb_raise( rb_eArgError, "Inputs rank should be 1, but got %d", na_input->rank );
  }

  if ( na_input->total != layer_ff->num_inputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer input size %d", na_input->total, layer_ff->num_inputs );
  }

  ////////////////////////////////////////////////////////////////////////////////////
  // Check target is valid
  val_target = na_cast_object(target, NA_SFLOAT);
  GetNArray( val_target, na_target );

  if ( na_target->rank != 1 ) {
    rb_raise( rb_eArgError, "Target rank should be 1, but got %d", na_target->rank );
  }

  layer_ff = p_network_last_layer_ff( network );

  if ( na_target->total != layer_ff->num_outputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match network output size %d", na_target->total, layer_ff->num_outputs );
  }

  ////////////////////////////////////////////////////////////////////////////////////
  // Run the training
  p_network_train_once( network, val_input, val_target );

  ////////////////////////////////////////////////////////////////////////////////////
  // Return nil
  return Qnil;
}

/* @overload learning_rate
 * @!attribute learning_rate
 * Multiplies weight adjustments due to error deltas during training. Range from 1e-6 to 1000.0
 * @return [Float]
 */
VALUE network_object_learning_rate( VALUE self ) {
  MLP_Network *network = get_network_struct( self );
  return FLT2NUM( network->eta );
}

/* @overload learning_rate=( new_learning_rate )
 * @!attribute learning_rate
 * Sets learning_rate.
 * @param [Float] new_learning_rate Range from 1e-6 to 1000.0.
 * @return [Float]
 */
VALUE network_object_set_learning_rate( VALUE self, VALUE new_learning_rate ) {
  float new_eta;
  MLP_Network *network = get_network_struct( self );

  new_eta = NUM2FLT( new_learning_rate );
  if ( new_eta < 1.0e-9 || new_eta > 1000.0 ) {
    rb_raise( rb_eArgError, "Learning rate %0.f out of bounds (0.000000001 to 1000.0)", new_eta );
  }

  network->eta = new_eta;

  return FLT2NUM( network->eta );
}

/* @overload momentum
 * @!attribute momentum
 * Multiplies weight adjustments due to previous adjustment during training. Range from 0.0 to 0.99
 * @return [Float]
 */
VALUE network_object_momentum( VALUE self ) {
  MLP_Network *network = get_network_struct( self );
  return FLT2NUM( network->momentum );
}

/* @overload momentum=( new_momentum )
 * @!attribute momentum
 * Sets momentum.
 * @param [Float] new_momentum Range from 0.0 to 0.99
 * @return [Float]
 */
VALUE network_object_set_momentum( VALUE self, VALUE val_momentum ) {
  float new_momentum;
  MLP_Network *network = get_network_struct( self );

  new_momentum = NUM2FLT( val_momentum );
  if ( new_momentum < 0.0 || new_momentum > 0.999 ) {
    rb_raise( rb_eArgError, "Momentum %0.6f out of bounds (0.0 to 0.99)", new_momentum );
  }

  network->momentum = new_momentum;
  return FLT2NUM( network->momentum );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_network_class( ) {
  // Network instantiation and class methods
  rb_define_alloc_func( RuNeNe_Network, network_alloc );
  rb_define_method( RuNeNe_Network, "initialize", network_class_initialize, 3 );
  rb_define_method( RuNeNe_Network, "initialize_copy", network_class_initialize_copy, 1 );

  // Network attributes
  rb_define_method( RuNeNe_Network, "num_layers", network_object_num_layers, 0 );
  rb_define_method( RuNeNe_Network, "num_inputs", network_object_num_inputs, 0 );
  rb_define_method( RuNeNe_Network, "num_outputs", network_object_num_outputs, 0 );
  rb_define_method( RuNeNe_Network, "input", network_object_input, 0 );
  rb_define_method( RuNeNe_Network, "output", network_object_output, 0 );
  rb_define_method( RuNeNe_Network, "layers", network_object_layers, 0 );
  rb_define_method( RuNeNe_Network, "learning_rate", network_object_learning_rate, 0 );
  rb_define_method( RuNeNe_Network, "momentum", network_object_momentum, 0 );
  rb_define_method( RuNeNe_Network, "learning_rate=", network_object_set_learning_rate, 1 );
  rb_define_method( RuNeNe_Network, "momentum=", network_object_set_momentum, 1 );

  // Network methods
  rb_define_method( RuNeNe_Network, "init_weights", network_object_init_weights, -1 );
  rb_define_method( RuNeNe_Network, "run", network_object_run, 1 );
  rb_define_method( RuNeNe_Network, "ms_error", network_object_ms_error, 1 );
  rb_define_method( RuNeNe_Network, "train_once", network_object_train_once, 2 );
  rb_define_singleton_method( RuNeNe_Network, "from_layer", network_class_from_layer, 1 );
}
