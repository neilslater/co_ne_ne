// ext/ru_ne_ne/ruby_class_nn_model.c

#include "ruby_class_nn_model.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_nn_model.c
//

inline VALUE nn_model_as_ruby_class( NNModel *nn_model , VALUE klass ) {
  return Data_Wrap_Struct( klass, nn_model__gc_mark, nn_model__destroy, nn_model );
}

VALUE nn_model_alloc(VALUE klass) {
  return nn_model_as_ruby_class( nn_model__create(), klass );
}

inline NNModel *get_nn_model_struct( VALUE obj ) {
  NNModel *nn_model;
  Data_Get_Struct( obj, NNModel, nn_model );
  return nn_model;
}

void assert_value_wraps_nn_model( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)nn_model__destroy) {
    rb_raise( rb_eTypeError, "Expected a NNModel object, but got something else" );
  }
}

// Converts anything compatile with layer description into layer object suitable
// for storing in nn_model
VALUE cast_nn_model_layer( volatile VALUE rv_layer_def, int *last_num_outputs ) {
  volatile VALUE this_layer;
  volatile VALUE rv_var;
  Layer_FF * layer_ff;
  int n_inputs = *last_num_outputs;

  if ( TYPE(rv_layer_def) == T_HASH ) {
    rv_var = ValAtSymbol( rv_layer_def, "num_inputs" );
    if ( !NIL_P(rv_var) ) {
      n_inputs = NUM2INT( rv_var  );
    }

    this_layer = layer_ff_new_ruby_object(
      n_inputs,
      NUM2INT( ValAtSymbol( rv_layer_def, "num_outputs" ) ),
      symbol_to_transfer_type( ValAtSymbol( rv_layer_def, "transfer" ) )
    );
  } else {
    this_layer = rv_layer_def;
    assert_value_wraps_layer_ff( this_layer );
  }

  Data_Get_Struct( this_layer, Layer_FF, layer_ff );
  *last_num_outputs = layer_ff->num_outputs;

  return this_layer;
}

/* Document-class: RuNeNe::NNModel
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  NNModel method definitions
//

/* @overload initialize( layers )
 * Creates a new NNModel
 * @param [Array<RuNeNe::Layer::Feedforward>] layers ...
 * @return [RuNeNe::NNModel] new ...
 */
VALUE nn_model_rbobject__initialize( VALUE self, VALUE rv_layers ) {
  NNModel *nn_model = get_nn_model_struct( self );
  // This stack-based var avoids memory leaks from alloc which might not be freed on error
  VALUE layers[100];
  int i, n, last_num_outputs = 0;

  Check_Type( rv_layers, T_ARRAY );

  n = FIX2INT( rb_funcall( rv_layers, rb_intern("count"), 0 ) );
  if ( n < 1 ) {
    rb_raise( rb_eArgError, "no layers in nn_model" );
  }
  if ( n > 100 ) {
    rb_raise( rb_eArgError, "too many layers in nn_model" );
  }

  for ( i = 0; i < n; i++ ) {
    layers[i] = cast_nn_model_layer( rb_ary_entry( rv_layers, i ), &last_num_outputs );
  }

  nn_model__init( nn_model, n, layers);

  return self;
}

/* @overload clone
 * When cloned, the returned NNModel has deep copies of C data.
 * @return [RuNeNe::NNModel] new
 */
VALUE nn_model_rbobject__initialize_copy( VALUE copy, VALUE orig ) {
  NNModel *nn_model_copy;
  NNModel *nn_model_orig;

  if (copy == orig) return copy;
  nn_model_orig = get_nn_model_struct( orig );
  nn_model_copy = get_nn_model_struct( copy );

  nn_model__deep_copy( nn_model_copy, nn_model_orig );

  return copy;
}

/* @!attribute [r] layers
 * Description goes here
 * @return [Array<RuNeNe::Layer::Feedforward>]]
 */
VALUE nn_model_rbobject__get_layers( VALUE self ) {
  NNModel *nn_model = get_nn_model_struct( self );
  int i;

  volatile VALUE rv_layers = rb_ary_new2( nn_model->num_layers );
  for ( i = 0; i < nn_model->num_layers; i++ ) {
    rb_ary_store( rv_layers, i, nn_model->layers[i] );
  }

  return rv_layers;
}

/* @overload layer( layer_id )
 * @param [Integer] layer_id index of layer
 * @return [RuNeNe::Layer::Feedforward]
 */
VALUE nn_model_rbobject__get_layer( VALUE self, VALUE rv_layer_id ) {
  NNModel *nn_model = get_nn_model_struct( self );
  int i = NUM2INT( rv_layer_id );

  if ( i < 0  || i >= nn_model->num_layers ) {
    rb_raise( rb_eArgError, "layer_id %d is out of bounds for this network (0..%d)", i, nn_model->num_layers - 1 );
  }

  return nn_model->layers[i];
}

/* @!attribute [r] num_layers
 * Description goes here
 * @return [Integer]
 */
VALUE nn_model_rbobject__get_num_layers( VALUE self ) {
  NNModel *nn_model = get_nn_model_struct( self );
  return INT2NUM( nn_model->num_layers );
}

/* @!attribute [r] num_inputs
 * Description goes here
 * @return [Integer]
 */
VALUE nn_model_rbobject__get_num_inputs( VALUE self ) {
  NNModel *nn_model = get_nn_model_struct( self );
  return INT2NUM( nn_model->num_inputs );
}

/* @!attribute [r] num_outputs
 * Description goes here
 * @return [Integer]
 */
VALUE nn_model_rbobject__get_num_outputs( VALUE self ) {
  NNModel *nn_model = get_nn_model_struct( self );
  return INT2NUM( nn_model->num_outputs );
}


/* @overload init_weights( mult = 1.0 )
 * Initialises weights in all layers.
 * @param [Float] mult optional size factor
 * @return [RuNeNe::NNModel] self
 */
VALUE nn_model_rbobject__init_weights( int argc, VALUE* argv, VALUE self ) {
  NNModel *nn_model = get_nn_model_struct( self );
  VALUE rv_mult;
  Layer_FF *layer_ff;
  float m = 1.0;
  int i, j, t;
  struct NARRAY *narr;

  rb_scan_args( argc, argv, "01", &rv_mult );
  if ( ! NIL_P( rv_mult ) ) {
    m = NUM2FLT( rv_mult );
  }

  for ( i = 0; i < nn_model->num_layers; i++ ) {
    // TODO: This only works for Layer_FF layers, we need a more flexible system
    Data_Get_Struct( nn_model->layers[i], Layer_FF, layer_ff );

    layer_ff__init_weights( layer_ff );

    if ( m != 0 ) {
      GetNArray( layer_ff->narr_weights, narr );
      t = narr->total;
      for ( j = 0; j < t; j++ ) {
        layer_ff->weights[j] *= m;
      }
    }
  }

  return self;
}

/* @overload run( input )
 * Runs nn_model forward and generates a result
 * @param [NArray<sfloat>] input single input vector
 * @return [NArray<sfloat>] output of nn_model
 */
VALUE nn_model_rbobject__run( VALUE self, VALUE rv_input ) {
  NNModel *nn_model = get_nn_model_struct( self );
  Layer_FF *layer_ff;
  int i;
  int out_shape[1] = { nn_model->num_outputs };

  struct NARRAY *na_input;
  volatile VALUE val_input = na_cast_object(rv_input, NA_SFLOAT);
  GetNArray( val_input, na_input );

  // Shouldn't happen, but we don't want a segfault
  if ( nn_model->num_layers < 1 ) {
    return Qnil;
  }

  if ( na_input->total != nn_model->num_inputs ) {
    rb_raise( rb_eArgError, "Input array must be size %d, but it was size %d", nn_model->num_inputs, na_input->total );
  }

  struct NARRAY *na_output;

  volatile VALUE val_output = na_make_object( NA_SFLOAT, 1, out_shape, cNArray );
  GetNArray( val_output, na_output );

  Data_Get_Struct( nn_model->layers[0], Layer_FF, layer_ff );
  layer_ff__run( layer_ff, (float*) na_input->ptr, nn_model->activations[0] );

  for ( i = 1; i < nn_model->num_layers; i++ ) {
    // TODO: This only works for Layer_FF layers, we need a more flexible system
    Data_Get_Struct( nn_model->layers[i], Layer_FF, layer_ff );
    layer_ff__run( layer_ff, nn_model->activations[i-1], nn_model->activations[i] );
  }

  memcpy( (float*) na_output->ptr, nn_model->activations[nn_model->num_layers-1], nn_model->num_outputs * sizeof(float) );

  return val_output;
}


/* @overload activations( layer_id )
 * Array of activation values from last call to .run from layer identified by layer_id
 * @param [NArray<sfloat>] input single input vector
 * @return [NArray<sfloat>] output of nn_model
 */
VALUE nn_model_rbobject__activations( VALUE self, VALUE rv_layer_id ) {
  NNModel *nn_model = get_nn_model_struct( self );
  Layer_FF *layer_ff;
  int layer_id = NUM2INT( rv_layer_id );

  if ( layer_id < 0 || layer_id >= nn_model->num_layers ) {
    return Qnil; // Should this raise instead? Not sure . . .
  }

  Data_Get_Struct( nn_model->layers[ layer_id ], Layer_FF, layer_ff );
  int out_shape[1] = { layer_ff->num_outputs };

  struct NARRAY *na_output;

  volatile VALUE val_output = na_make_object( NA_SFLOAT, 1, out_shape, cNArray );
  GetNArray( val_output, na_output );

  memcpy( (float*) na_output->ptr, nn_model->activations[layer_id], layer_ff->num_outputs * sizeof(float) );

  return val_output;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void init_nn_model_class( ) {
  // NNModel instantiation and class methods
  rb_define_alloc_func( RuNeNe_NNModel, nn_model_alloc );
  rb_define_method( RuNeNe_NNModel, "initialize", nn_model_rbobject__initialize, 1 );
  rb_define_method( RuNeNe_NNModel, "initialize_copy", nn_model_rbobject__initialize_copy, 1 );

  // NNModel attributes
  rb_define_method( RuNeNe_NNModel, "layers", nn_model_rbobject__get_layers, 0 );
  rb_define_method( RuNeNe_NNModel, "num_layers", nn_model_rbobject__get_num_layers, 0 );
  rb_define_method( RuNeNe_NNModel, "num_inputs", nn_model_rbobject__get_num_inputs, 0 );
  rb_define_method( RuNeNe_NNModel, "num_outputs", nn_model_rbobject__get_num_outputs, 0 );

  // NNModel methods
  rb_define_method( RuNeNe_NNModel, "layer", nn_model_rbobject__get_layer, 1 );
  rb_define_method( RuNeNe_NNModel, "init_weights", nn_model_rbobject__init_weights, -1 );
  rb_define_method( RuNeNe_NNModel, "run", nn_model_rbobject__run, 1 );
  rb_define_method( RuNeNe_NNModel, "activations", nn_model_rbobject__activations, 1 );
}
