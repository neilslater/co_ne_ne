// ext/co_ne_ne/ruby_class_training_data.c

#include "ruby_class_training_data.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_training_data.c
//

VALUE RuNeNe_TrainingData = Qnil;

inline VALUE training_data_as_ruby_class( TrainingData *training_data , VALUE klass ) {
  return Data_Wrap_Struct( klass, p_training_data_gc_mark, p_training_data_destroy, training_data );
}

VALUE training_data_alloc(VALUE klass) {
  return training_data_as_ruby_class( p_training_data_create(), klass );
}

inline TrainingData *get_training_data_struct( VALUE obj ) {
  TrainingData *training_data;
  Data_Get_Struct( obj, TrainingData, training_data );
  return training_data;
}

void assert_value_wraps_training_data( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)p_training_data_destroy) {
    rb_raise( rb_eTypeError, "Expected a TrainingData object, but got something else" );
  }
}

/* Document-class:  RuNeNe::TrainingData
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
 * @return [RuNeNe::TrainingData] new network consisting of new layers, with random weights
 */
VALUE training_data_class_initialize( VALUE self, VALUE inputs, VALUE targets ) {
  volatile VALUE val_inputs;
  volatile VALUE val_targets;
  struct NARRAY *na_inputs;
  struct NARRAY *na_targets;
  TrainingData *training_data = get_training_data_struct( self );

  val_inputs = na_cast_object( inputs, NA_SFLOAT );
  GetNArray( val_inputs, na_inputs );

  val_targets = na_cast_object( targets, NA_SFLOAT );
  GetNArray( val_targets, na_targets );

  if ( na_inputs->rank < 2 ) {
    rb_raise( rb_eArgError, "Inputs rank should be at least 2, but got %d", na_inputs->rank );
  }

  if ( na_targets->rank < 2 ) {
    rb_raise( rb_eArgError, "Targets rank should be at least 2, but got %d", na_targets->rank );
  }

  if ( na_inputs->shape[ na_inputs->rank - 1 ] != na_targets->shape[ na_targets->rank - 1 ] ) {
    rb_raise( rb_eArgError, "Number of input items %d not same as target items %d",
        na_inputs->shape[ na_inputs->rank - 1 ], na_targets->shape[ na_targets->rank - 1 ] );
  }

  p_training_data_init_from_narray( training_data, val_inputs, val_targets );

  return self;
}

/* @overload clone
 * When cloned, the returned TrainingData has deep copies of inputs and outputs,
 * @return [RuNeNe::TrainingData] new training data with identical items to caller.
 */
VALUE training_data_class_initialize_copy( VALUE copy, VALUE orig ) {
  TrainingData *training_data_copy;
  TrainingData *training_data_orig;

  if (copy == orig) return copy;
  training_data_orig = get_training_data_struct( orig );
  training_data_copy = get_training_data_struct( copy );

  training_data_copy->num_items = training_data_orig->num_items;
  training_data_copy->narr_outputs = na_clone( training_data_orig->narr_outputs );
  training_data_copy->narr_inputs = na_clone( training_data_orig->narr_inputs );

  return copy;
}

/* @!attribute [r] inputs
 * The inputs array.
 * @return [NArray<sfloat>]
 */
VALUE training_data_object_inputs( VALUE self ) {
  TrainingData *training_data = get_training_data_struct( self );
  return training_data->narr_inputs;
}

/* @!attribute [r] outputs
 * The outputs array.
 * @return [NArray<sfloat>]
 */
VALUE training_data_object_outputs( VALUE self ) {
  TrainingData *training_data = get_training_data_struct( self );
  return training_data->narr_outputs;
}

/* @!attribute [r] num_items
 * The number of training items.
 * @return [Integer]
 */
VALUE training_data_object_num_items( VALUE self ) {
  TrainingData *training_data = get_training_data_struct( self );
  return INT2NUM( training_data->num_items );
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void init_training_data_class( ) {
  volatile VALUE conene_root = rb_define_module( "RuNeNe" );
  RuNeNe_TrainingData = rb_define_class_under( conene_root, "TrainingData", rb_cObject );

  // TrainingData instantiation and class methods
  rb_define_alloc_func( RuNeNe_TrainingData, training_data_alloc );
  rb_define_method( RuNeNe_TrainingData, "initialize", training_data_class_initialize, 2 );
  rb_define_method( RuNeNe_TrainingData, "initialize_copy", training_data_class_initialize_copy, 1 );

  // TrainingData attributes
  rb_define_method( RuNeNe_TrainingData, "inputs", training_data_object_inputs, 0 );
  rb_define_method( RuNeNe_TrainingData, "outputs", training_data_object_outputs, 0 );
  rb_define_method( RuNeNe_TrainingData, "num_items", training_data_object_num_items, 0 );
}
