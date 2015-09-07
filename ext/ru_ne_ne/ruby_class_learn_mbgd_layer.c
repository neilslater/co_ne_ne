// ext/ru_ne_ne/ruby_class_learn_mbgd_layer.c

#include "ruby_class_learn_mbgd_layer.h"

// Helper for converting hash to C properties
void copy_hash_to_mbgd_layer_properties( VALUE rv_opts, MBGDLayer *mbgd_layer ) {
  volatile VALUE rv_var;
  volatile VALUE new_narray;
  struct NARRAY* narr;
  float momentum = 0.9, decay = 0.9, epsilon = 1e-6;
  GradientDescent_SGD * gd_sgd;
  GradientDescent_NAG * gd_nag;
  GradientDescent_RMSProp * gd_rmsprop;

  // Start with simple properties
  rv_var = ValAtSymbol(rv_opts,"learning_rate");
  if ( !NIL_P(rv_var) ) {
    mbgd_layer->learning_rate = NUM2FLT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"weight_decay");
  if ( !NIL_P(rv_var) ) {
    mbgd_layer->weight_decay = NUM2FLT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"max_norm");
  if ( !NIL_P(rv_var) ) {
    mbgd_layer->max_norm = NUM2FLT( rv_var );
  }

  // Now deal with more complex properties, allow setting of NArrays, provided they fit

  rv_var = ValAtSymbol(rv_opts,"de_dz");
  if ( !NIL_P(rv_var) ) {
    new_narray = na_cast_object(rv_var, NA_SFLOAT);
    GetNArray( new_narray, narr );
    if ( narr->rank != 1 ) {
      rb_raise( rb_eArgError, "de_dz rank should be 1, but got %d", narr->rank );
    }

    if ( narr->shape[0] !=  mbgd_layer->num_outputs ) {
      rb_raise( rb_eArgError, "de_dz size %d is not same as num_outputs %d",narr->shape[0], mbgd_layer->num_outputs );
    }
    mbgd_layer->narr_de_dz = new_narray;
    mbgd_layer->de_dz = (float *) narr->ptr;
  }

  rv_var = ValAtSymbol(rv_opts,"de_da");
  if ( !NIL_P(rv_var) ) {
    new_narray = na_cast_object(rv_var, NA_SFLOAT);
    GetNArray( new_narray, narr );
    if ( narr->rank != 1 ) {
      rb_raise( rb_eArgError, "de_da rank should be 1, but got %d", narr->rank );
    }

    if ( narr->shape[0] != mbgd_layer->num_inputs ) {
      rb_raise( rb_eArgError, "de_dz size %d is not same as num_inputs %d",narr->shape[0], mbgd_layer->num_inputs );
    }
    mbgd_layer->narr_de_da = new_narray;
    mbgd_layer->de_da = (float *) narr->ptr;
  }

  rv_var = ValAtSymbol(rv_opts,"de_dw");
  if ( !NIL_P(rv_var) ) {
    new_narray = na_cast_object(rv_var, NA_SFLOAT);
    GetNArray( new_narray, narr );
    if ( narr->rank != 2 ) {
      rb_raise( rb_eArgError, "de_dw rank should be 2, but got %d", narr->rank );
    }

    if ( narr->shape[0] != ( 1 + mbgd_layer->num_inputs ) ) {
      rb_raise( rb_eArgError, "de_dw num columns %d is not same as (num_inputs+1) = %d",narr->shape[0], mbgd_layer->num_inputs + 1 );
    }

    if ( narr->shape[1] != ( mbgd_layer->num_outputs ) ) {
      rb_raise( rb_eArgError, "de_dw num rows %d is not same as num_outputs %d",narr->shape[0], mbgd_layer->num_outputs );
    }
    mbgd_layer->narr_de_dw = new_narray;
    mbgd_layer->de_dw = (float *) narr->ptr;
  }

  rv_var = ValAtSymbol(rv_opts,"gd_optimiser");
  if ( !NIL_P(rv_var) ) {
    int t = ( 1 + mbgd_layer->num_inputs ) * mbgd_layer->num_outputs;

    if ( TYPE(rv_var) != T_DATA ) {
      rb_raise( rb_eTypeError, "Expected a GradientDescent object for :gd_optimiser, but got something else" );
    }

    if ( RDATA(rv_var)->dfree == (RUBY_DATA_FUNC)gd_sgd__destroy ) {
      Data_Get_Struct( rv_var, GradientDescent_SGD, gd_sgd );
      if ( gd_sgd->num_params != t ) {
        rb_raise( rb_eArgError, "Supplied GradientDescent object is set for %d params, but need %d", gd_sgd->num_params, t  );
      }
      mbgd_layer->gd_accel_type = GDACCEL_TYPE_NONE;
    } else if ( RDATA(rv_var)->dfree == (RUBY_DATA_FUNC)gd_nag__destroy ) {
      Data_Get_Struct( rv_var, GradientDescent_NAG, gd_nag );
      if ( gd_nag->num_params != t ) {
        rb_raise( rb_eArgError, "Supplied GradientDescent object is set for %d params, but need %d", gd_nag->num_params, t  );
      }
      mbgd_layer->gd_accel_type = GDACCEL_TYPE_MOMENTUM;
    } else if ( RDATA(rv_var)->dfree == (RUBY_DATA_FUNC)gd_rmsprop__destroy ) {
      Data_Get_Struct( rv_var, GradientDescent_RMSProp, gd_rmsprop );
      if ( gd_rmsprop->num_params != t ) {
        rb_raise( rb_eArgError, "Supplied GradientDescent object is set for %d params, but need %d", gd_rmsprop->num_params, t  );
      }
      mbgd_layer->gd_accel_type = GDACCEL_TYPE_RMSPROP;
    } else {
      rb_raise( rb_eTypeError, "Expected a GradientDescent object for :gd_optimiser, but got something else" );
    }
    mbgd_layer->gd_optimiser = rv_var;
  } else {
    rv_var = ValAtSymbol(rv_opts,"momentum");
    if ( !NIL_P(rv_var) ) {
      momentum = NUM2FLT( rv_var );
    }

    rv_var = ValAtSymbol(rv_opts,"decay");
    if ( !NIL_P(rv_var) ) {
      decay = NUM2FLT( rv_var );
    }

    rv_var = ValAtSymbol(rv_opts,"epsilon");
    if ( !NIL_P(rv_var) ) {
      epsilon = NUM2FLT( rv_var );
    }

    mbgd_layer__init_gd_optimiser( mbgd_layer,
        symbol_to_gd_accel_type( ValAtSymbol(rv_opts, "gd_accel_type") ),
        momentum, decay, epsilon );
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_mbgd_layer.c
//

inline VALUE mbgd_layer_as_ruby_class( MBGDLayer *mbgd_layer , VALUE klass ) {
  return Data_Wrap_Struct( klass, mbgd_layer__gc_mark, mbgd_layer__destroy, mbgd_layer );
}

VALUE mbgd_layer_alloc(VALUE klass) {
  return mbgd_layer_as_ruby_class( mbgd_layer__create(), klass );
}

inline MBGDLayer *get_mbgd_layer_struct( VALUE obj ) {
  MBGDLayer *mbgd_layer;
  Data_Get_Struct( obj, MBGDLayer, mbgd_layer );
  return mbgd_layer;
}

void assert_value_wraps_mbgd_layer( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)mbgd_layer__destroy) {
    rb_raise( rb_eTypeError, "Expected a MBGDLayer object, but got something else" );
  }
}

/* Document-class:  RuNeNe::Learn::MBGD::Layer
 *
 * This class models the training algorithms and data used across a single layer during gradient
 * descent by backpropagation. An instance of this class represents the training state of a specific
 * layer in a network.
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Network method definitions
//

/* @overload initialize( opts )
 * Creates a new RuNeNe::Learn::MBGD::Layer instance. In normal use, the network trainer will create
 * the necessary layer objects automatically from the network acrhitecture.
 * @param [Hash] opts initialisation options
 * @return [RuNeNe::Learn::MBGD::Layer] the new RuNeNe::Learn::MBGD::Layer object.
 */
VALUE mbgd_layer_rbobject__initialize( VALUE self, VALUE rv_opts ) {
  volatile VALUE rv_var;
  int num_ins, num_outs;
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );

  Check_Type( rv_opts, T_HASH );

  rv_var = ValAtSymbol(rv_opts,"num_inputs");
  if ( NIL_P(rv_var) ) {
    rb_raise( rb_eArgError, "Missing :num_inputs option" );
  }
  num_ins = NUM2INT( rv_var );
  if ( num_ins < 1 ) {
    rb_raise( rb_eArgError, "Input size %d is less than minimum of 1", num_ins );
  }

  rv_var = ValAtSymbol(rv_opts,"num_outputs");
  if ( NIL_P(rv_var) ) {
    rb_raise( rb_eArgError, "Missing :num_outputs option" );
  }
  num_outs = NUM2INT( rv_var );
  if ( num_outs  < 1 ) {
    rb_raise( rb_eArgError, "Output size %d is less than minimum of 1", num_outs );
  }

  mbgd_layer__init( mbgd_layer, num_ins, num_outs );

  copy_hash_to_mbgd_layer_properties( rv_opts, mbgd_layer );

  return self;
}

/* @overload from_layer( opts )
 * Creates a new RuNeNe::Learn::MBGD::Layer instance to match a given layer
 * @param [RuNeNe::Layer::FeedForward] layer to create training structures for
 * @param [Hash] opts initialisation options
 * @return [RuNeNe::Learn::MBGD::Layer] the new RuNeNe::Learn::MBGD::Layer object.
 */
VALUE mbgd_layer_rbclass__from_layer( int argc, VALUE* argv, VALUE self ) {
  volatile VALUE rv_layer, rv_opts;
  Layer_FF *layer_ff;
  MBGDLayer *mbgd_layer;

  rb_scan_args( argc, argv, "11", &rv_layer, &rv_opts );

  // Check we really have a layer object to build on
  if ( TYPE(rv_layer) != T_DATA ||
      RDATA(rv_layer)->dfree != (RUBY_DATA_FUNC)layer_ff__destroy) {
    rb_raise( rb_eTypeError, "Expected a Layer object, but got something else" );
  }
  Data_Get_Struct( rv_layer, Layer_FF, layer_ff );

  if (!NIL_P(rv_opts)) {
    Check_Type( rv_opts, T_HASH );
  }

  volatile VALUE rv_new_mbgd_layer = mbgd_layer_alloc( RuNeNe_Learn_MBGD_Layer );
  mbgd_layer = get_mbgd_layer_struct( rv_new_mbgd_layer );

  mbgd_layer__init( mbgd_layer, layer_ff->num_inputs, layer_ff->num_outputs );

  if (!NIL_P(rv_opts)) {
    copy_hash_to_mbgd_layer_properties( rv_opts, mbgd_layer );
  }

  return rv_new_mbgd_layer;
}

/* @overload clone
 * When cloned, the returned MBGDLayer has deep copies of C data.
 * @return [RuNeNe::Learn::MBGD::Layer] new
 */
VALUE mbgd_layer_rbobject__initialize_copy( VALUE copy, VALUE orig ) {
  MBGDLayer *mbgd_layer_copy;
  MBGDLayer *mbgd_layer_orig;

  if (copy == orig) return copy;
  mbgd_layer_orig = get_mbgd_layer_struct( orig );
  mbgd_layer_copy = get_mbgd_layer_struct( copy );

  mbgd_layer__deep_copy( mbgd_layer_copy, mbgd_layer_orig );

  return copy;
}

/* @!attribute [r] num_inputs
 * Description goes here
 * @return [Integer]
 */
VALUE mbgd_layer_rbobject__get_num_inputs( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return INT2NUM( mbgd_layer->num_inputs );
}

/* @!attribute [r] num_outputs
 * Description goes here
 * @return [Integer]
 */
VALUE mbgd_layer_rbobject__get_num_outputs( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return INT2NUM( mbgd_layer->num_outputs );
}

/* @!attribute learning_rate
 * Description goes here
 * @return [Float]
 */
VALUE mbgd_layer_rbobject__get_learning_rate( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return FLT2NUM( mbgd_layer->learning_rate );
}

VALUE mbgd_layer_rbobject__set_learning_rate( VALUE self, VALUE rv_learning_rate ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  mbgd_layer->learning_rate = NUM2FLT( rv_learning_rate );
  return rv_learning_rate;
}

/* @!attribute [r] gd_accel_type
 * Description goes here
 * @return [Integer]
 */
VALUE mbgd_layer_rbobject__get_gd_accel_type( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return gd_accel_type_to_symbol( mbgd_layer->gd_accel_type );
}

VALUE mbgd_layer_rbobject__set_gd_accel_type( VALUE self, VALUE rv_gd_accel_type ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );

  mbgd_layer->gd_accel_type = symbol_to_gd_accel_type( rv_gd_accel_type );

  return rv_gd_accel_type;
}

/* @!attribute max_norm
 * Description goes here
 * @return [Float]
 */
VALUE mbgd_layer_rbobject__get_max_norm( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return FLT2NUM( mbgd_layer->max_norm );
}

VALUE mbgd_layer_rbobject__set_max_norm( VALUE self, VALUE rv_max_norm ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  mbgd_layer->max_norm = NUM2FLT( rv_max_norm );
  return rv_max_norm;
}

/* @!attribute weight_decay
 * Description goes here
 * @return [Float]
 */
VALUE mbgd_layer_rbobject__get_weight_decay( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return FLT2NUM( mbgd_layer->weight_decay );
}

VALUE mbgd_layer_rbobject__set_weight_decay( VALUE self, VALUE rv_weight_decay ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  mbgd_layer->weight_decay = NUM2FLT( rv_weight_decay );
  return rv_weight_decay;
}

/* @!attribute  [r] de_dz
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE mbgd_layer_rbobject__get_narr_de_dz( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return mbgd_layer->narr_de_dz;
}

/* @!attribute  [r] de_da
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE mbgd_layer_rbobject__get_narr_de_da( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return mbgd_layer->narr_de_da;
}

/* @!attribute  [r] de_dw
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE mbgd_layer_rbobject__get_narr_de_dw( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return mbgd_layer->narr_de_dw;
}

/* @!attribute gd_optimiser
 * Description goes here
 * @return [RuNeNe::GradientDescent::SGD,RuNeNe::GradientDescent::NAG,RuNeNe::GradientDescent::RMSProp]
 */
VALUE mbgd_layer_rbobject__get_gd_optimiser( VALUE self ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  return mbgd_layer->gd_optimiser;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Backprop methods
//

/* @overload start_batch( layer )
 * Description goes here
 * @param [RuNeNe::Layer::FeedForward] layer
 * @return [NArray<sfloat>] self
 */
VALUE mbgd_layer_rbobject__start_batch( VALUE self, VALUE rv_layer ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  Layer_FF *layer_ff;

  // Check we really have a layer object to fetch output from
  if ( TYPE(rv_layer) != T_DATA ||
      RDATA(rv_layer)->dfree != (RUBY_DATA_FUNC)layer_ff__destroy) {
    rb_raise( rb_eTypeError, "Expected a Layer object, but got something else" );
  }
  Data_Get_Struct( rv_layer, Layer_FF, layer_ff );
  // TODO: Check layer is same size as trainer

  mbgd_layer__start_batch( mbgd_layer, layer_ff );
  return self;
}

/* @overload backprop_for_output_layer( layer, input, output, target, objective_type )
 * Calculates the partial derivative of objective function with respect to layer z values, given
 * current layer outputs and the target values it is expected to learn. Sets the value of de_dz
 * internally.
 * @param [RuNeNe::Layer::FeedForward] layer
 * @param [NArray<sfloat>] input
 * @param [NArray<sfloat>] output
 * @param [NArray<sfloat>] target
 * @param [Symbol] objective
 * @return [RuNeNe::Learn::MBGD::Layer] self
 */

VALUE mbgd_layer_rbobject__backprop_for_output_layer( VALUE self, VALUE rv_layer, VALUE rv_input, VALUE rv_output, VALUE rv_target, VALUE rv_objective ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  Layer_FF *layer_ff;
  objective_type o = symbol_to_objective_type( rv_objective );
  struct NARRAY* narr_target;
  struct NARRAY* narr_output;
  struct NARRAY* narr_input;
  volatile VALUE target_narray;
  volatile VALUE output_narray;
  volatile VALUE input_narray;

  // Check we really have a layer object to fetch output from
  if ( TYPE(rv_layer) != T_DATA ||
      RDATA(rv_layer)->dfree != (RUBY_DATA_FUNC)layer_ff__destroy) {
    rb_raise( rb_eTypeError, "Expected a Layer object, but got something else" );
  }
  Data_Get_Struct( rv_layer, Layer_FF, layer_ff );

  if ( layer_ff->num_outputs != mbgd_layer->num_outputs ) {
    rb_raise( rb_eArgError, "layer has %d outputs, but trainer is expecting %d", layer_ff->num_outputs, mbgd_layer->num_outputs );
  }

  // Validate inputs array is correct size
  input_narray = na_cast_object(rv_input, NA_SFLOAT);
  GetNArray( input_narray, narr_input );
  if ( narr_input->rank != 1 ) {
    rb_raise( rb_eArgError, "input rank should be 1, but got %d", narr_input->rank );
  }

  if ( narr_input->shape[0] != mbgd_layer->num_inputs ) {
    rb_raise( rb_eArgError, "input has %d entries, but trainer is expecting %d", narr_input->shape[0], mbgd_layer->num_inputs );
  }

  // Validate targets array is correct size
  target_narray = na_cast_object(rv_target, NA_SFLOAT);
  GetNArray( target_narray, narr_target );
  if ( narr_target->rank != 1 ) {
    rb_raise( rb_eArgError, "target rank should be 1, but got %d", narr_target->rank );
  }

  if ( narr_target->shape[0] != mbgd_layer->num_outputs ) {
    rb_raise( rb_eArgError, "target has %d entries, but trainer is expecting %d", narr_target->shape[0], mbgd_layer->num_outputs );
  }

  // Validate outputs array is correct size
  output_narray = na_cast_object(rv_output, NA_SFLOAT);
  GetNArray( output_narray, narr_output );
  if ( narr_output->rank != 1 ) {
    rb_raise( rb_eArgError, "output rank should be 1, but got %d", narr_output->rank );
  }

  if ( narr_output->shape[0] != mbgd_layer->num_outputs ) {
    rb_raise( rb_eArgError, "output has %d entries, but trainer is expecting %d", narr_output->shape[0], mbgd_layer->num_outputs );
  }

  mbgd_layer__backprop_for_output_layer( mbgd_layer, layer_ff,
      (float *) narr_input->ptr,  (float *) narr_output->ptr, (float *) narr_target->ptr, o );
  return self;
}


/* @overload backprop_for_mid_layer( layer, input, output, upper_de_da )
 * Calculates the partial derivative of objective function with respect to layer z values, given
 * current layer outputs and the target values it is expected to learn. Sets the value of de_dz
 * internally.
 * @param [RuNeNe::Layer::FeedForward] layer
 * @param [NArray<sfloat>] input
 * @param [NArray<sfloat>] output
 * @param [NArray<sfloat>] upper_de_da
 * @return [RuNeNe::Learn::MBGD::Layer] self
 */

VALUE mbgd_layer_rbobject__backprop_for_mid_layer( VALUE self, VALUE rv_layer, VALUE rv_input, VALUE rv_output, VALUE rv_upper_de_da, VALUE rv_objective ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  Layer_FF *layer_ff;

  struct NARRAY* narr_upper_de_da;
  struct NARRAY* narr_output;
  struct NARRAY* narr_input;
  volatile VALUE upper_de_da_narray;
  volatile VALUE output_narray;
  volatile VALUE input_narray;

  // Check we really have a layer object to fetch output from
  if ( TYPE(rv_layer) != T_DATA ||
      RDATA(rv_layer)->dfree != (RUBY_DATA_FUNC)layer_ff__destroy) {
    rb_raise( rb_eTypeError, "Expected a Layer object, but got something else" );
  }
  Data_Get_Struct( rv_layer, Layer_FF, layer_ff );

  if ( layer_ff->num_outputs != mbgd_layer->num_outputs ) {
    rb_raise( rb_eArgError, "layer has %d outputs, but trainer is expecting %d", layer_ff->num_outputs, mbgd_layer->num_outputs );
  }

  // Validate inputs array is correct size
  input_narray = na_cast_object(rv_input, NA_SFLOAT);
  GetNArray( input_narray, narr_input );
  if ( narr_input->rank != 1 ) {
    rb_raise( rb_eArgError, "input rank should be 1, but got %d", narr_input->rank );
  }

  if ( narr_input->shape[0] != mbgd_layer->num_inputs ) {
    rb_raise( rb_eArgError, "input has %d entries, but trainer is expecting %d", narr_input->shape[0], mbgd_layer->num_inputs );
  }

  // Validate upper_de_das array is correct size
  upper_de_da_narray = na_cast_object(rv_upper_de_da, NA_SFLOAT);
  GetNArray( upper_de_da_narray, narr_upper_de_da );
  if ( narr_upper_de_da->rank != 1 ) {
    rb_raise( rb_eArgError, "upper_de_da rank should be 1, but got %d", narr_upper_de_da->rank );
  }

  if ( narr_upper_de_da->shape[0] != mbgd_layer->num_outputs ) {
    rb_raise( rb_eArgError, "upper_de_da has %d entries, but trainer is expecting %d", narr_upper_de_da->shape[0], mbgd_layer->num_outputs );
  }

  // Validate outputs array is correct size
  output_narray = na_cast_object(rv_output, NA_SFLOAT);
  GetNArray( output_narray, narr_output );
  if ( narr_output->rank != 1 ) {
    rb_raise( rb_eArgError, "output rank should be 1, but got %d", narr_output->rank );
  }

  if ( narr_output->shape[0] != mbgd_layer->num_outputs ) {
    rb_raise( rb_eArgError, "output has %d entries, but trainer is expecting %d", narr_output->shape[0], mbgd_layer->num_outputs );
  }

  mbgd_layer__backprop_for_mid_layer( mbgd_layer, layer_ff,
      (float *) narr_input->ptr,  (float *) narr_output->ptr, (float *) narr_upper_de_da->ptr );

  return self;
}


/* @overload finish_batch( layer )
 * Finishes up current batch by modifying weights in layer
 * @param [RuNeNe::Layer::FeedForward] layer
 * @return [RuNeNe::Learn::MBGD::Layer] self
 */

VALUE mbgd_layer_rbobject__finish_batch( VALUE self, VALUE rv_layer ) {
  MBGDLayer *mbgd_layer = get_mbgd_layer_struct( self );
  Layer_FF *layer_ff;

  // Check we really have a layer object to fetch output from
  if ( TYPE(rv_layer) != T_DATA ||
      RDATA(rv_layer)->dfree != (RUBY_DATA_FUNC)layer_ff__destroy) {
    rb_raise( rb_eTypeError, "Expected a Layer object, but got something else" );
  }
  Data_Get_Struct( rv_layer, Layer_FF, layer_ff );
  mbgd_layer__finish_batch( mbgd_layer, layer_ff );

  return self;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mbgd_layer_class( ) {
  // MBGDLayer instantiation and class methods
  rb_define_alloc_func( RuNeNe_Learn_MBGD_Layer, mbgd_layer_alloc );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "initialize", mbgd_layer_rbobject__initialize, 1 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "initialize_copy", mbgd_layer_rbobject__initialize_copy, 1 );
  rb_define_singleton_method( RuNeNe_Learn_MBGD_Layer, "from_layer", mbgd_layer_rbclass__from_layer, -1 );

  // MBGDLayer attributes
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "num_inputs", mbgd_layer_rbobject__get_num_inputs, 0 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "num_outputs", mbgd_layer_rbobject__get_num_outputs, 0 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "de_dz", mbgd_layer_rbobject__get_narr_de_dz, 0 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "de_da", mbgd_layer_rbobject__get_narr_de_da, 0 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "de_dw", mbgd_layer_rbobject__get_narr_de_dw, 0 );

  rb_define_method( RuNeNe_Learn_MBGD_Layer, "learning_rate", mbgd_layer_rbobject__get_learning_rate, 0 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "learning_rate=", mbgd_layer_rbobject__set_learning_rate, 1 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "gd_accel_type", mbgd_layer_rbobject__get_gd_accel_type, 0 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "gd_accel_type=", mbgd_layer_rbobject__set_gd_accel_type, 1 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "gd_optimiser", mbgd_layer_rbobject__get_gd_optimiser, 0 );

  rb_define_method( RuNeNe_Learn_MBGD_Layer, "max_norm", mbgd_layer_rbobject__get_max_norm, 0 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "max_norm=", mbgd_layer_rbobject__set_max_norm, 1 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "weight_decay", mbgd_layer_rbobject__get_weight_decay, 0 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "weight_decay=", mbgd_layer_rbobject__set_weight_decay, 1 );

  // MBGDLayer methods
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "start_batch", mbgd_layer_rbobject__start_batch, 1 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "backprop_for_output_layer", mbgd_layer_rbobject__backprop_for_output_layer, 5 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "backprop_for_mid_layer", mbgd_layer_rbobject__backprop_for_mid_layer, 4 );
  rb_define_method( RuNeNe_Learn_MBGD_Layer, "finish_batch", mbgd_layer_rbobject__finish_batch, 1 );
}
