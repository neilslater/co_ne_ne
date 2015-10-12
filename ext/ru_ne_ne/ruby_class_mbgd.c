// ext/ru_ne_ne/ruby_class_mbgd.c

#include "ruby_class_mbgd.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_mbgd.c
//

inline VALUE mbgd_as_ruby_class( MBGD *mbgd , VALUE klass ) {
  return Data_Wrap_Struct( klass, mbgd__gc_mark, mbgd__destroy, mbgd );
}

VALUE mbgd_alloc(VALUE klass) {
  return mbgd_as_ruby_class( mbgd__create(), klass );
}

inline MBGD *get_mbgd_struct( VALUE obj ) {
  MBGD *mbgd;
  Data_Get_Struct( obj, MBGD, mbgd );
  return mbgd;
}

void assert_value_wraps_mbgd( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)mbgd__destroy) {
    rb_raise( rb_eTypeError, "Expected a MBGD object, but got something else" );
  }
}

// Converts anything compatile with layer description into layer object suitable
// for storing in mbgd
VALUE cast_mbgd_layer( volatile VALUE rv_layer_def, int *last_num_outputs ) {
  volatile VALUE this_layer;
  volatile VALUE rv_var;
  MBGDLayer * mbgd_layer;
  int n_inputs = *last_num_outputs;
  int n_outputs = 0;

  if ( TYPE(rv_layer_def) == T_HASH ) {
    mbgd_layer = mbgd_layer__create();

    rv_var = ValAtSymbol( rv_layer_def, "num_inputs" );
    if ( !NIL_P(rv_var) ) {
      n_inputs = NUM2INT( rv_var  );
    }

    mbgd_layer__init( mbgd_layer, n_inputs, NUM2INT( ValAtSymbol( rv_layer_def, "num_outputs" ) ) );
    copy_hash_to_mbgd_layer_properties( rv_layer_def, mbgd_layer, 1 );
    this_layer = Data_Wrap_Struct( RuNeNe_Learn_MBGD_Layer, mbgd_layer__gc_mark, mbgd_layer__destroy, mbgd_layer );
  } else {
    this_layer = rv_layer_def;
    assert_value_wraps_mbgd_layer( this_layer );
  }

  Data_Get_Struct( this_layer, MBGDLayer, mbgd_layer );
  *last_num_outputs = mbgd_layer->num_outputs;

  return this_layer;
}

// We cannot accept per-layer state when processing all-layers hash
void check_opts_hash_for_mbgd_layer_state( VALUE rv_opts ) {
  if ( !NIL_P( ValAtSymbol(rv_opts,"de_dz") ) ) {
    rb_raise( rb_eArgError, "de_dz not allowed across whole network" );
  }

  if ( !NIL_P( ValAtSymbol(rv_opts,"de_da") ) ) {
    rb_raise( rb_eArgError, "de_da not allowed across whole network" );
  }

  if ( !NIL_P( ValAtSymbol(rv_opts,"de_dw") ) ) {
    rb_raise( rb_eArgError, "de_dw not allowed across whole network" );
  }

  if ( !NIL_P( ValAtSymbol(rv_opts,"gradient_descent") ) ) {
    rb_raise( rb_eArgError, "gradient_descent not allowed across whole network" );
  }

  return;
}

/* Document-class: RuNeNe::Learn::MBGD
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Network method definitions
//

/* @overload initialize( mbgd_layers )
 * Creates a new ...
 * @param [Array<RuNeNe::Learn::MBGD::Layer>] mbgd_layers ...
 * @return [RuNeNe::Learn::MBGD] new ...
 */
VALUE mbgd_rbobject__initialize( VALUE self, VALUE rv_mbgd_layers, VALUE rv_objective_type ) {
  MBGD *mbgd = get_mbgd_struct( self );
  objective_type o = symbol_to_objective_type( rv_objective_type );

  // This stack-based var avoids memory leaks from alloc which might not be freed on error
  VALUE layers[100];
  int i, n, last_num_outputs = 0;

  Check_Type( rv_mbgd_layers, T_ARRAY );

  n = FIX2INT( rb_funcall( rv_mbgd_layers, rb_intern("count"), 0 ) );
  if ( n < 1 ) {
    rb_raise( rb_eArgError, "no layers in mbgd" );
  }
  if ( n > 100 ) {
    rb_raise( rb_eArgError, "too many layers in mbgd" );
  }

  for ( i = 0; i < n; i++ ) {
    layers[i] = cast_mbgd_layer( rb_ary_entry( rv_mbgd_layers, i ), &last_num_outputs );
  }

  mbgd__init( mbgd, n, layers, o );

  return self;
}

/* @overload clone
 * When cloned, the returned MBGD has deep copies of C data.
 * @return [RuNeNe::Learn::MBGD] new
 */
VALUE mbgd_rbobject__initialize_copy( VALUE copy, VALUE orig ) {
  MBGD *mbgd_copy;
  MBGD *mbgd_orig;

  if (copy == orig) return copy;
  mbgd_orig = get_mbgd_struct( orig );
  mbgd_copy = get_mbgd_struct( copy );

  mbgd__deep_copy( mbgd_copy, mbgd_orig );

  return copy;
}

/* @overload from_nn_model( nn_model, opts )
 * Creates a new RuNeNe::Learn::MBGD instance to match a given NNModel
 * @param [RuNeNe::NNModel] nn_model to create training structures for
 * @param [Hash] opts initialisation options, applied to all MBGD layers
 * @return [RuNeNe::Learn::MBGD] the new RuNeNe::Learn::MBGD object.
 */
VALUE mbgd_layer_rbclass__from_nn_model( int argc, VALUE* argv, VALUE self ) {
  volatile VALUE rv_nn_model, rv_opts, mbgd_layer_args[2], mbgd_layers[100];
  NNModel *nn_model;
  MBGD *mbgd;
  int i;
  objective_type o = MSE;

  rb_scan_args( argc, argv, "11", &rv_nn_model, &rv_opts );

  // Check we really have a nn_model object to build from
  if ( TYPE(rv_nn_model) != T_DATA ||
      RDATA(rv_nn_model)->dfree != (RUBY_DATA_FUNC)nn_model__destroy) {
    rb_raise( rb_eTypeError, "Expected a NNModel object, but got something else" );
  }
  Data_Get_Struct( rv_nn_model, NNModel, nn_model );

  if (!NIL_P(rv_opts)) {
    Check_Type( rv_opts, T_HASH );
    check_opts_hash_for_mbgd_layer_state( rv_opts );

    if ( !NIL_P( ValAtSymbol(rv_opts,"objective") ) ) {
      o = symbol_to_objective_type( ValAtSymbol(rv_opts,"objective") );
    }
  }

  volatile VALUE rv_new_mbgd = mbgd_alloc( RuNeNe_Learn_MBGD );
  mbgd = get_mbgd_struct( rv_new_mbgd );

  for( i = 0; i < nn_model->num_layers; i++ ) {
    mbgd_layer_args[0] = nn_model->layers[i];
    mbgd_layer_args[1] = rv_opts;
    mbgd_layers[i] = mbgd_layer_rbclass__from_layer( 2, (VALUE*)mbgd_layer_args, RuNeNe_Learn_MBGD_Layer );
  }

  mbgd__init( mbgd, nn_model->num_layers, (VALUE*)mbgd_layers, o );

  return rv_new_mbgd;
}

/* @!attribute [r] mbgd_layers
 * Description goes here
 * @return [Object]
 */
VALUE mbgd_rbobject__get_mbgd_layers( VALUE self ) {
  MBGD *mbgd = get_mbgd_struct( self );

  int i;

  volatile VALUE rv_layers = rb_ary_new2( mbgd->num_layers );
  for ( i = 0; i < mbgd->num_layers; i++ ) {
    rb_ary_store( rv_layers, i, mbgd->mbgd_layers[i] );
  }

  return rv_layers;
}

/* @overload layer( layer_id )
 * @param [Integer] layer_id index of layer
 * @return [RuNeNe::Learn::MBGD::Layer]
 */
VALUE mbgd_rbobject__get_layer( VALUE self, VALUE rv_layer_id ) {
  MBGD *mbgd = get_mbgd_struct( self );
  int i = NUM2INT( rv_layer_id );

  if ( i < 0  || i >= mbgd->num_layers ) {
    rb_raise( rb_eArgError, "layer_id %d is out of bounds for this network (0..%d)", i, mbgd->num_layers - 1 );
  }

  return mbgd->mbgd_layers[i];
}

/* @overload set_meta_params( opts )
 * @param [Hash] opts hash of properties to change and their new values
 * @return [RuNeNe::Learn::MBGD] self
 */
VALUE mbgd_rbobject__set_meta_params( VALUE self, VALUE rv_opts ) {
  MBGD *mbgd = get_mbgd_struct( self );
  MBGDLayer *mbgd_layer;
  int i;

  Check_Type( rv_opts, T_HASH );
  check_opts_hash_for_mbgd_layer_state( rv_opts );
  if ( !NIL_P( ValAtSymbol(rv_opts,"gradient_descent_type") ) ) {
    rb_raise( rb_eArgError, "gradient_descent_type cannot be changed in an existing learner" );
  }

  for ( i = 0; i < mbgd->num_layers; i++ ) {
    mbgd_layer = mbgd__get_mbgd_layer_at( mbgd, i );
    copy_hash_to_mbgd_layer_properties( rv_opts, mbgd_layer, 0 );
  }

  return self;
}

/* @overload train_one_batch( nn_model, dataset, objective, batch_size )
 * @param [RuNeNe::NNModel] nn_model network architecture to be trained
 * @param [RuNeNe::Dataset] dataset training data
 * @param [Symbol] objective objective function to optimise (TODO: Should be set in constructor)
 * @param [Integer] batch_size number of items in dataset to process before altering weights
 * @return [Float] mean score of objective function for batch
 */
VALUE mbgd_rbobject__train_one_batch( VALUE self, VALUE rv_nn_model, VALUE rv_dataset, VALUE rv_batch_size ) {
  MBGD *mbgd = get_mbgd_struct( self );
  NNModel *nn_model = safe_get_nn_model_struct( rv_nn_model );
  DataSet *dataset = safe_get_dataset_struct( rv_dataset );

  int batch_size = NUM2INT( rv_batch_size );

  mbgd__check_size_compatible( mbgd, nn_model, dataset );
  return FLT2NUM( mbgd__train_one_batch( mbgd, nn_model, dataset, batch_size ) );
}


/* @!attribute [r] num_layers
 * Description goes here
 * @return [Integer]
 */
VALUE mbgd_rbobject__get_num_layers( VALUE self ) {
  MBGD *mbgd = get_mbgd_struct( self );
  return INT2NUM( mbgd->num_layers );
}

/* @!attribute [r] num_inputs
 * Description goes here
 * @return [Integer]
 */
VALUE mbgd_rbobject__get_num_inputs( VALUE self ) {
  MBGD *mbgd = get_mbgd_struct( self );
  return INT2NUM( mbgd->num_inputs );
}

/* @!attribute [r] num_outputs
 * Description goes here
 * @return [Integer]
 */
VALUE mbgd_rbobject__get_num_outputs( VALUE self ) {
  MBGD *mbgd = get_mbgd_struct( self );
  return INT2NUM( mbgd->num_outputs );
}

/* @!attribute [r] objective
 * Description goes here
 * @return [Module]
 */
VALUE mbgd_rbobject__get_objective( VALUE self ) {
  MBGD *mbgd = get_mbgd_struct( self );
  return objective_type_to_module( mbgd->objective );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mbgd_class( ) {
  // MBGD instantiation and class methods
  rb_define_alloc_func( RuNeNe_Learn_MBGD, mbgd_alloc );
  rb_define_method( RuNeNe_Learn_MBGD, "initialize", mbgd_rbobject__initialize, 2 );
  rb_define_method( RuNeNe_Learn_MBGD, "initialize_copy", mbgd_rbobject__initialize_copy, 1 );
  rb_define_singleton_method( RuNeNe_Learn_MBGD, "from_nn_model", mbgd_layer_rbclass__from_nn_model, -1 );

  // MBGD attributes
  rb_define_method( RuNeNe_Learn_MBGD, "mbgd_layers", mbgd_rbobject__get_mbgd_layers, 0 );
  rb_define_method( RuNeNe_Learn_MBGD, "num_layers", mbgd_rbobject__get_num_layers, 0 );
  rb_define_method( RuNeNe_Learn_MBGD, "num_inputs", mbgd_rbobject__get_num_inputs, 0 );
  rb_define_method( RuNeNe_Learn_MBGD, "num_outputs", mbgd_rbobject__get_num_outputs, 0 );
  rb_define_method( RuNeNe_Learn_MBGD, "objective", mbgd_rbobject__get_objective, 0 );

  // MBGD methods
  rb_define_method( RuNeNe_Learn_MBGD, "layer", mbgd_rbobject__get_layer, 1 );
  rb_define_method( RuNeNe_Learn_MBGD, "set_meta_params", mbgd_rbobject__set_meta_params, 1 );
  rb_define_method( RuNeNe_Learn_MBGD, "train_one_batch", mbgd_rbobject__train_one_batch, 3 );
}
