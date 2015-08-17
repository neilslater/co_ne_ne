// ext/ru_ne_ne/ruby_class_trainer_bp_layer.c

#include "ruby_class_trainer_bp_layer.h"

// Hash lookup helper
VALUE ValAtSymbol(VALUE hash, const char* key) { return rb_hash_lookup(hash, ID2SYM( rb_intern(key) ) ); }

// Convert symbol to internal smoothing type
bp_smooth_type smoothing_type_from_symbol( VALUE smth_type ) {
  ID smth_id;

  smth_id = rb_intern("none");
  if ( ! NIL_P(smth_type) ) {
    if ( TYPE(smth_type) != T_SYMBOL ) {
      rb_raise( rb_eTypeError, "Expected symbol for smoothing type" );
    }
    smth_id = SYM2ID(smth_type);
  }

  if ( rb_intern("none") == smth_id ) {
    return SMOOTH_TYPE_NONE;
  } else if ( rb_intern("momentum") == smth_id ) {
    return SMOOTH_TYPE_MOMENTUM;
  } else if ( rb_intern("rmsprop") == smth_id ) {
    return SMOOTH_TYPE_RMSPROP;
  } else {
    rb_raise( rb_eArgError, "Smoothing type %s not recognised", rb_id2name(smth_id) );
  }
}

// Helper for converting hash to C properties
void copy_hash_to_bplayer_properties( VALUE rv_opts, TrainerBPLayer *trainer_bp_layer ) {
  volatile VALUE rv_var;

  rv_var = ValAtSymbol(rv_opts,"learning_rate");
  if ( !NIL_P(rv_var) ) {
    trainer_bp_layer->learning_rate = NUM2FLT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"smoothing_rate");
  if ( !NIL_P(rv_var) ) {
    trainer_bp_layer->smoothing_rate = NUM2FLT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"weight_decay");
  if ( !NIL_P(rv_var) ) {
    trainer_bp_layer->weight_decay = NUM2FLT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"max_norm");
  if ( !NIL_P(rv_var) ) {
    trainer_bp_layer->max_norm = NUM2FLT( rv_var );
  }

  trainer_bp_layer->smoothing_type = smoothing_type_from_symbol( ValAtSymbol(rv_opts,"smoothing_type") );

  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_trainer_bp_layer.c
//

inline VALUE trainer_bp_layer_as_ruby_class( TrainerBPLayer *trainer_bp_layer , VALUE klass ) {
  return Data_Wrap_Struct( klass, trainer_bp_layer__gc_mark, trainer_bp_layer__destroy, trainer_bp_layer );
}

VALUE trainer_bp_layer_alloc(VALUE klass) {
  return trainer_bp_layer_as_ruby_class( trainer_bp_layer__create(), klass );
}

inline TrainerBPLayer *get_trainer_bp_layer_struct( VALUE obj ) {
  TrainerBPLayer *trainer_bp_layer;
  Data_Get_Struct( obj, TrainerBPLayer, trainer_bp_layer );
  return trainer_bp_layer;
}

void assert_value_wraps_trainer_bp_layer( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)trainer_bp_layer__destroy) {
    rb_raise( rb_eTypeError, "Expected a TrainerBPLayer object, but got something else" );
  }
}

/* Document-class:  RuNeNe::Trainer::BPLayer
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
 * Creates a new RuNeNe::Trainer::BPLayer instance. In normal use, the network trainer will create
 * the necessary layer objects automatically from the network acrhitecture.
 * @param [Hash] opts initialisation options
 * @return [RuNeNe::Trainer::BPLayer] the new RuNeNe::Trainer::BPLayer object.
 */
VALUE trainer_bp_layer_rbobject__initialize( VALUE self, VALUE rv_opts ) {
  volatile VALUE rv_var;
  int num_ins, num_outs;
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );

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

  trainer_bp_layer__init( trainer_bp_layer, num_ins, num_outs );

  copy_hash_to_bplayer_properties( rv_opts, trainer_bp_layer );

  return self;
}

/* @overload clone
 * When cloned, the returned TrainerBPLayer has deep copies of C data.
 * @return [RuNeNe::Trainer::BPLayer] new
 */
VALUE trainer_bp_layer_rbobject__initialize_copy( VALUE copy, VALUE orig ) {
  TrainerBPLayer *trainer_bp_layer_copy;
  TrainerBPLayer *trainer_bp_layer_orig;

  if (copy == orig) return copy;
  trainer_bp_layer_orig = get_trainer_bp_layer_struct( orig );
  trainer_bp_layer_copy = get_trainer_bp_layer_struct( copy );

  trainer_bp_layer__deep_copy( trainer_bp_layer_copy, trainer_bp_layer_orig );

  return copy;
}

/* @!attribute [r] num_inputs
 * Description goes here
 * @return [Integer]
 */
VALUE trainer_bp_layer_rbobject__get_num_inputs( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return INT2NUM( trainer_bp_layer->num_inputs );
}

/* @!attribute [r] num_outputs
 * Description goes here
 * @return [Integer]
 */
VALUE trainer_bp_layer_rbobject__get_num_outputs( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return INT2NUM( trainer_bp_layer->num_outputs );
}

/* @!attribute learning_rate
 * Description goes here
 * @return [Float]
 */
VALUE trainer_bp_layer_rbobject__get_learning_rate( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return FLT2NUM( trainer_bp_layer->learning_rate );
}

VALUE trainer_bp_layer_rbobject__set_learning_rate( VALUE self, VALUE rv_learning_rate ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  trainer_bp_layer->learning_rate = NUM2FLT( rv_learning_rate );
  return rv_learning_rate;
}

/* @!attribute [r] smoothing_type
 * Description goes here
 * @return [Integer]
 */
VALUE trainer_bp_layer_rbobject__get_smoothing_type( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );

  switch( trainer_bp_layer->smoothing_type ) {
    case SMOOTH_TYPE_NONE:
      return ID2SYM( rb_intern("none") );
    case SMOOTH_TYPE_MOMENTUM:
      return ID2SYM( rb_intern("momentum") );
    case SMOOTH_TYPE_RMSPROP:
      return ID2SYM( rb_intern("rmsprop") );
    default:
      rb_raise( rb_eRuntimeError, "smoothing_type not valid, internal error");
  }
}

/* @!attribute smoothing_rate
 * Description goes here
 * @return [Float]
 */
VALUE trainer_bp_layer_rbobject__get_smoothing_rate( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return FLT2NUM( trainer_bp_layer->smoothing_rate );
}

VALUE trainer_bp_layer_rbobject__set_smoothing_rate( VALUE self, VALUE rv_smoothing_rate ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  trainer_bp_layer->smoothing_rate = NUM2FLT( rv_smoothing_rate );
  return rv_smoothing_rate;
}

/* @!attribute max_norm
 * Description goes here
 * @return [Float]
 */
VALUE trainer_bp_layer_rbobject__get_max_norm( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return FLT2NUM( trainer_bp_layer->max_norm );
}

VALUE trainer_bp_layer_rbobject__set_max_norm( VALUE self, VALUE rv_max_norm ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  trainer_bp_layer->max_norm = NUM2FLT( rv_max_norm );
  return rv_max_norm;
}

/* @!attribute weight_decay
 * Description goes here
 * @return [Float]
 */
VALUE trainer_bp_layer_rbobject__get_weight_decay( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return FLT2NUM( trainer_bp_layer->weight_decay );
}

VALUE trainer_bp_layer_rbobject__set_weight_decay( VALUE self, VALUE rv_weight_decay ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  trainer_bp_layer->weight_decay = NUM2FLT( rv_weight_decay );
  return rv_weight_decay;
}

/* @!attribute  [r] de_dz
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE trainer_bp_layer_rbobject__get_narr_de_dz( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return trainer_bp_layer->narr_de_dz;
}

/* @!attribute  [r] de_da
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE trainer_bp_layer_rbobject__get_narr_de_da( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return trainer_bp_layer->narr_de_da;
}

/* @!attribute  [r] de_dw
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE trainer_bp_layer_rbobject__get_narr_de_dw( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return trainer_bp_layer->narr_de_dw;
}

/* @!attribute  [r] de_dw_momentum
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE trainer_bp_layer_rbobject__get_narr_de_dw_momentum( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return trainer_bp_layer->narr_de_dw_momentum;
}

/* @!attribute  [r] de_dw_rmsprop
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE trainer_bp_layer_rbobject__get_narr_de_dw_rmsprop( VALUE self ) {
  TrainerBPLayer *trainer_bp_layer = get_trainer_bp_layer_struct( self );
  return trainer_bp_layer->narr_de_dw_rmsprop;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_trainer_bp_layer_class( ) {
  // TrainerBPLayer instantiation and class methods
  rb_define_alloc_func( RuNeNe_Trainer_BPLayer, trainer_bp_layer_alloc );
  rb_define_method( RuNeNe_Trainer_BPLayer, "initialize", trainer_bp_layer_rbobject__initialize, 1 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "initialize_copy", trainer_bp_layer_rbobject__initialize_copy, 1 );

  // TrainerBPLayer attributes
  rb_define_method( RuNeNe_Trainer_BPLayer, "num_inputs", trainer_bp_layer_rbobject__get_num_inputs, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "num_outputs", trainer_bp_layer_rbobject__get_num_outputs, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "de_dz", trainer_bp_layer_rbobject__get_narr_de_dz, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "de_da", trainer_bp_layer_rbobject__get_narr_de_da, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "de_dw", trainer_bp_layer_rbobject__get_narr_de_dw, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "de_dw_momentum", trainer_bp_layer_rbobject__get_narr_de_dw_momentum, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "de_dw_rmsprop", trainer_bp_layer_rbobject__get_narr_de_dw_rmsprop, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "learning_rate", trainer_bp_layer_rbobject__get_learning_rate, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "learning_rate=", trainer_bp_layer_rbobject__set_learning_rate, 1 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "smoothing_type", trainer_bp_layer_rbobject__get_smoothing_type, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "smoothing_rate", trainer_bp_layer_rbobject__get_smoothing_rate, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "smoothing_rate=", trainer_bp_layer_rbobject__set_smoothing_rate, 1 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "max_norm", trainer_bp_layer_rbobject__get_max_norm, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "max_norm=", trainer_bp_layer_rbobject__set_max_norm, 1 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "weight_decay", trainer_bp_layer_rbobject__get_weight_decay, 0 );
  rb_define_method( RuNeNe_Trainer_BPLayer, "weight_decay=", trainer_bp_layer_rbobject__set_weight_decay, 1 );
}
