// ext/ru_ne_ne/ruby_class_dataset.c

#include "ruby_class_dataset.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_dataset.c
//

inline VALUE dataset_as_ruby_class( DataSet *dataset , VALUE klass ) {
  return Data_Wrap_Struct( klass, dataset__gc_mark, dataset__destroy, dataset );
}

VALUE dataset_alloc(VALUE klass) {
  return dataset_as_ruby_class( dataset__create(), klass );
}

inline DataSet *get_dataset_struct( VALUE obj ) {
  DataSet *dataset;
  Data_Get_Struct( obj, DataSet, dataset );
  return dataset;
}

void assert_value_wraps_dataset( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)dataset__destroy) {
    rb_raise( rb_eTypeError, "Expected a DataSet object, but got something else" );
  }
}

/* Document-class:  RuNeNe::DataSet
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
 * @return [RuNeNe::DataSet] new network consisting of new layers, with random weights
 */
VALUE dataset_class_initialize( VALUE self, VALUE rv_inputs, VALUE rv_targets ) {
  volatile VALUE val_inputs;
  volatile VALUE val_targets;
  struct NARRAY *na_inputs;
  struct NARRAY *na_targets;
  DataSet *dataset = get_dataset_struct( self );

  val_inputs = na_cast_object( rv_inputs, NA_SFLOAT );
  GetNArray( val_inputs, na_inputs );

  val_targets = na_cast_object( rv_targets, NA_SFLOAT );
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

  dataset__init_from_narray( dataset, val_inputs, val_targets );

  return self;
}

/* @overload clone
 * When cloned, the returned DataSet has deep copies of inputs and outputs,
 * @return [RuNeNe::DataSet] new training data with identical items to caller.
 */
VALUE dataset_class_initialize_copy( VALUE copy, VALUE orig ) {
  DataSet *dataset_copy;
  DataSet *dataset_orig;

  if (copy == orig) return copy;
  dataset_orig = get_dataset_struct( orig );
  dataset_copy = get_dataset_struct( copy );

  dataset_copy->num_items = dataset_orig->num_items;
  dataset_copy->narr_outputs = na_clone( dataset_orig->narr_outputs );
  dataset_copy->narr_inputs = na_clone( dataset_orig->narr_inputs );

  dataset__reinit( dataset_copy );

  return copy;
}

/* @!attribute [r] inputs
 * The inputs array.
 * @return [NArray<sfloat>]
 */
VALUE dataset_object_inputs( VALUE self ) {
  DataSet *dataset = get_dataset_struct( self );
  return dataset->narr_inputs;
}

/* @!attribute [r] outputs
 * The outputs array.
 * @return [NArray<sfloat>]
 */
VALUE dataset_object_outputs( VALUE self ) {
  DataSet *dataset = get_dataset_struct( self );
  return dataset->narr_outputs;
}

/* @!attribute [r] num_items
 * The number of training items.
 * @return [Integer]
 */
VALUE dataset_object_num_items( VALUE self ) {
  DataSet *dataset = get_dataset_struct( self );
  return INT2NUM( dataset->num_items );
}

VALUE dataset_object_next_item( VALUE self ) {
  dataset__next( get_dataset_struct( self ) );
  return self;
}

VALUE dataset_object_current_input_item( VALUE self ) {
  DataSet *dataset = get_dataset_struct( self );
  float *input_data = dataset__current_input( dataset );

  struct NARRAY *narr;
  volatile VALUE current_input = na_make_object( NA_SFLOAT, dataset->input_item_rank, dataset->input_item_shape, cNArray );
  GetNArray( current_input, narr );
  memcpy( (float*) narr->ptr, input_data, dataset->input_item_size * sizeof(float) );

  return current_input;
}

VALUE dataset_object_current_output_item( VALUE self ) {
  DataSet *dataset = get_dataset_struct( self );
  float *output_data = dataset__current_output( dataset );

  struct NARRAY *narr;
  volatile VALUE current_output = na_make_object( NA_SFLOAT, dataset->output_item_rank, dataset->output_item_shape, cNArray );
  GetNArray( current_output, narr );
  memcpy( (float*) narr->ptr, output_data, dataset->output_item_size * sizeof(float) );

  return current_output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_dataset_class( ) {
  // DataSet instantiation and class methods
  rb_define_alloc_func( RuNeNe_DataSet, dataset_alloc );
  rb_define_method( RuNeNe_DataSet, "initialize", dataset_class_initialize, 2 );
  rb_define_method( RuNeNe_DataSet, "initialize_copy", dataset_class_initialize_copy, 1 );

  // DataSet attributes
  rb_define_method( RuNeNe_DataSet, "inputs", dataset_object_inputs, 0 );
  rb_define_method( RuNeNe_DataSet, "outputs", dataset_object_outputs, 0 );
  rb_define_method( RuNeNe_DataSet, "num_items", dataset_object_num_items, 0 );

  // Methods
  rb_define_method( RuNeNe_DataSet, "next_item", dataset_object_next_item, 0 );
  rb_define_method( RuNeNe_DataSet, "current_input_item", dataset_object_current_input_item, 0 );
  rb_define_method( RuNeNe_DataSet, "current_output_item", dataset_object_current_output_item, 0 );
}
