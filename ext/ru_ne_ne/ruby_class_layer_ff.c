// ext/ru_ne_ne/ruby_class_layer_ff.c

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for a single feed-forward layer - the deeper implementation is in
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
  layer_ff__init_weights( layer_ff );

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
  layer_ff__set_weights( layer_ff_copy, na_clone( layer_ff_orig->narr_weights ) );

  return copy;
}

void assert_value_wraps_layer_ff( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)layer_ff__destroy) {
    rb_raise( rb_eTypeError, "Expected a Layer object, but got something else" );
  }
}

void set_symbol_to_transfer_type( Layer_FF *layer_ff , VALUE tfn_type ) {
  layer_ff->transfer_fn = symbol_to_transfer_type( tfn_type );
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
  layer_ff__set_weights( layer_ff, weights );

  return rv_layer_ff;
}


/* Document-class:  RuNeNe::Layer::FeedForward
 *
 * An object of this class represents a layer in a feed-forward network.
 *
 * A general rule for using NArray parameters with this class is that *sfloat* NArrays
 * are used directly, and other types are cast to that type. This means that using
 * *sfloat* sub-type to manage weights is generally more efficient.
 */

//////////////////////////////////////////////////////////////////////////////////////
//
//  Layer method definitions
//

/* @overload initialize( num_inputs, num_outputs, transfer_label = :sigmoid )
 * Creates a new layer and randomly initializes the weights.
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

  set_symbol_to_transfer_type( layer_ff, tfn_type );

  layer_ff__new_narrays( layer_ff );
  layer_ff__init_weights( layer_ff );

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

  layer_ff__set_weights( layer_ff_copy, na_clone( layer_ff_orig->narr_weights ) );

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
  volatile VALUE weights_in, tfn_type;
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

  return layer_ff_new_ruby_object_from_weights( val_weights, symbol_to_transfer_type( tfn_type ) );
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
  return transfer_type_to_module( layer_ff->transfer_fn );
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

/* @overload init_weights( mult = 1.0 )
 * Initialises weights to a normal distribution based on number of inputs and outputs.
 * @param [Float] mult optional size factor
 * @return [RuNeNe::Layer::FeedForward] self
 */
VALUE layer_ff_object_init_weights( int argc, VALUE* argv, VALUE self ) {
  VALUE rv_mult;
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  double m = 1.0;
  int i, t;
  struct NARRAY *narr;

  rb_scan_args( argc, argv, "01", &rv_mult );

  layer_ff__init_weights( layer_ff );

  if ( ! NIL_P( rv_mult ) ) {
    m = NUM2DBL( rv_mult );
    GetNArray( layer_ff->narr_weights, narr );
    t = narr->total;
    for ( i = 0; i < t; i++ ) {
      layer_ff->weights[i] *= m;
    }
  }

  return self;
}


/* @overload run( )
 * Runs the layer with supplied input(s). The input array can be a single, one-dimensional
 * vector, or can be
 * @param [NArray<sfloat>] inputs
 * @return [NArray<sfloat>]
 */
VALUE layer_ff_object_run( VALUE self, VALUE rv_input ) {
  Layer_FF *layer_ff = get_layer_ff_struct( self );
  int out_shape[1] = { layer_ff->num_outputs };

  struct NARRAY *na_input;
  volatile VALUE val_input = na_cast_object(rv_input, NA_SFLOAT);
  GetNArray( val_input, na_input );

  if ( na_input->total != layer_ff->num_inputs ) {
    rb_raise( rb_eArgError, "Input array must be size %d, but it was size %d", layer_ff->num_inputs, na_input->total );
  }

  struct NARRAY *na_output;
  volatile VALUE val_output = na_make_object( NA_SFLOAT, 1, out_shape, cNArray );
  GetNArray( val_output, na_output );

  layer_ff__run( layer_ff, (float*) na_input->ptr, (float*) na_output->ptr );

  return val_output;
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
  rb_define_method( RuNeNe_Layer_FeedForward, "weights", layer_ff_object_weights, 0 );

  // FeedForward methods
  rb_define_method( RuNeNe_Layer_FeedForward, "init_weights", layer_ff_object_init_weights, -1 );
  rb_define_method( RuNeNe_Layer_FeedForward, "run", layer_ff_object_run, 1 );
}
