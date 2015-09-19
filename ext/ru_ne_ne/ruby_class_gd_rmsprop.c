// ext/ru_ne_ne/ruby_class_gd_rmsprop.c

#include "ruby_class_gd_rmsprop.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_gd_rmsprop.c
//

inline VALUE gd_rmsprop_as_ruby_class( GradientDescent_RMSProp *gd_rmsprop , VALUE klass ) {
  return Data_Wrap_Struct( klass, gd_rmsprop__gc_mark, gd_rmsprop__destroy, gd_rmsprop );
}

VALUE gd_rmsprop_alloc(VALUE klass) {
  return gd_rmsprop_as_ruby_class( gd_rmsprop__create(), klass );
}

inline GradientDescent_RMSProp *get_gd_rmsprop_struct( VALUE obj ) {
  GradientDescent_RMSProp *gd_rmsprop;
  Data_Get_Struct( obj, GradientDescent_RMSProp, gd_rmsprop );
  return gd_rmsprop;
}

void assert_value_wraps_gd_rmsprop( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)gd_rmsprop__destroy) {
    rb_raise( rb_eTypeError, "Expected a GradientDescent_RMSProp object, but got something else" );
  }
}

// Helper for converting hash to C properties
void copy_hash_to_gd_rmsprop_properties( VALUE rv_opts, GradientDescent_RMSProp *gd_rmsprop ) {
  volatile VALUE rv_var;
  volatile VALUE new_narray;
  struct NARRAY* narr;

  // Start with simple properties
  rv_var = ValAtSymbol(rv_opts,"num_params");
  if ( !NIL_P(rv_var) ) {
    gd_rmsprop->num_params = NUM2INT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"decay");
  if ( !NIL_P(rv_var) ) {
    gd_rmsprop->decay = NUM2FLT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"epsilon");
  if ( !NIL_P(rv_var) ) {
    gd_rmsprop->epsilon = NUM2FLT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"av_squared_grads");
  if ( !NIL_P(rv_var) ) {
    new_narray = na_cast_object(rv_var, NA_SFLOAT);
    GetNArray( new_narray, narr );
    gd_rmsprop->narr_av_squared_grads = new_narray;
    gd_rmsprop->av_squared_grads = (float *) narr->ptr;
    gd_rmsprop->num_params = narr->total;
  }

  // TODO: Deal with partially-complete object here, and detect
  // inconsistent params

  return;
}

/* Document-class: RuNeNe::GradientDescent::RMSProp
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  NNModel method definitions
//

/* @overload initialize( num_params, decay, epsilon )
 * Creates a new ...
 * @param [NArray<sfloat>] params actual or example NArray of params for optimisation
 * @param [Float] decay ...
 * @param [Float] epsilon ...
 * @return [RuNeNe::GradientDescent::RMSProp] new ...
 */

VALUE gd_rmsprop_rbobject__initialize( VALUE self, VALUE rv_params, VALUE rv_decay, VALUE rv_epsilon ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );

  volatile VALUE example_params = na_cast_object( rv_params, NA_SFLOAT );

  gd_rmsprop__init( gd_rmsprop, example_params, NUM2FLT( rv_decay ), NUM2FLT( rv_epsilon ) );

  return self;
}

/* @overload clone
 * When cloned, the returned GradientDescent_RMSProp has deep copies of C data.
 * @return [RuNeNe::GradientDescent::RMSProp] new
 */
VALUE gd_rmsprop_rbobject__initialize_copy( VALUE copy, VALUE orig ) {
  GradientDescent_RMSProp *gd_rmsprop_copy;
  GradientDescent_RMSProp *gd_rmsprop_orig;

  if (copy == orig) return copy;
  gd_rmsprop_orig = get_gd_rmsprop_struct( orig );
  gd_rmsprop_copy = get_gd_rmsprop_struct( copy );

  gd_rmsprop__deep_copy( gd_rmsprop_copy, gd_rmsprop_orig );

  return copy;
}

/* @overload initialize( h )
 * Creates a new ...
 * keys are h[:param_update_velocity], h[:nag]
 * @return [RuNeNe::GradientDescent::RMSProp] new ...
 */

VALUE gd_rmsprop_rbclass__from_h( VALUE self, VALUE rv_h ) {
  GradientDescent_RMSProp *gd_rmsprop;
  Check_Type( rv_h, T_HASH );

  VALUE rv_gd_rmsprop = gd_rmsprop_alloc( RuNeNe_GradientDescent_RMSProp );
  gd_rmsprop = get_gd_rmsprop_struct( rv_gd_rmsprop );

  copy_hash_to_gd_rmsprop_properties( rv_h, gd_rmsprop );

  return rv_gd_rmsprop;
}

/* @!attribute [r] num_params
 * Description goes here
 * @return [Integer]
 */
VALUE gd_rmsprop_rbobject__get_num_params( VALUE self ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );
  return INT2NUM( gd_rmsprop->num_params );
}

/* @!attribute decay
 * Description goes here
 * @return [Float]
 */
VALUE gd_rmsprop_rbobject__get_decay( VALUE self ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );
  return FLT2NUM( gd_rmsprop->decay );
}

VALUE gd_rmsprop_rbobject__set_decay( VALUE self, VALUE rv_decay ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );
  gd_rmsprop->decay = NUM2FLT( rv_decay );
  return rv_decay;
}

/* @!attribute epsilon
 * Description goes here
 * @return [Float]
 */
VALUE gd_rmsprop_rbobject__get_epsilon( VALUE self ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );
  return FLT2NUM( gd_rmsprop->epsilon );
}

VALUE gd_rmsprop_rbobject__set_epsilon( VALUE self, VALUE rv_epsilon ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );
  gd_rmsprop->epsilon = NUM2FLT( rv_epsilon );
  return rv_epsilon;
}

/* @!attribute [r] av_squared_grads
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE gd_rmsprop_rbobject__get_narr_av_squared_grads( VALUE self ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );
  return gd_rmsprop->narr_av_squared_grads;
}

/* @overload pre_gradient_step( params, learning_rate )
 * Prepares object for a gradient step. Some optimisers alter params
 * @param [NArray<sfloat>] params array of same size as initial example
 * @param [Float] learning_rate size of
 * @return [NArray<sfloat>] the params array that willbe optimised (may be cast to NArray<sfloat> from supplied params)
 */
VALUE gd_rmsprop_rbobject__pre_gradient_step( VALUE self, VALUE rv_params, VALUE rv_learning_rate ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );

  volatile VALUE opt_params;
  struct NARRAY *na_params;
  opt_params = na_cast_object( rv_params, NA_SFLOAT );
  GetNArray( opt_params, na_params );

  if ( gd_rmsprop->num_params != na_params->total ) {
    rb_raise( rb_eArgError, "Expecting NArray with %d params, but input has %d params", gd_rmsprop->num_params, na_params->total );
  }

  gd_rmsprop__pre_gradient_step( gd_rmsprop, (float *)na_params->ptr, NUM2FLT(rv_learning_rate) );

  return opt_params;
}

/* @overload pre_gradient_step( params, gradients, learning_rate )
 * Prepares object for a gradient step. Some optimisers alter params
 * @param [NArray<sfloat>] params array of same size as initial example
 * @param [NArray<sfloat>] gradients array of same size as initial example
 * @param [Float] learning_rate size of
 * @return [NArray<sfloat>] the params array that willbe optimised (may be cast to NArray<sfloat> from supplied params)
 */
VALUE gd_rmsprop_rbobject__gradient_step( VALUE self, VALUE rv_params, VALUE rv_gradients, VALUE rv_learning_rate ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );

  volatile VALUE opt_params;
  struct NARRAY *na_params;
  volatile VALUE gradients;
  struct NARRAY *na_grads;

  opt_params = na_cast_object( rv_params, NA_SFLOAT );
  GetNArray( opt_params, na_params );

  if ( gd_rmsprop->num_params != na_params->total ) {
    rb_raise( rb_eArgError, "Expecting NArray with %d params, but input has %d params", gd_rmsprop->num_params, na_params->total );
  }

  gradients = na_cast_object( rv_gradients, NA_SFLOAT );
  GetNArray( gradients, na_grads );

  if ( gd_rmsprop->num_params != na_grads->total ) {
    rb_raise( rb_eArgError, "Expecting NArray with %d params, but gradient has %d params", gd_rmsprop->num_params, na_grads->total );
  }

  gd_rmsprop__gradient_step( gd_rmsprop, (float *)na_params->ptr, (float *)na_grads->ptr, NUM2FLT(rv_learning_rate) );

  return opt_params;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_gd_rmsprop_class( ) {
  // GradientDescent_RMSProp instantiation and class methods
  rb_define_alloc_func( RuNeNe_GradientDescent_RMSProp, gd_rmsprop_alloc );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "initialize", gd_rmsprop_rbobject__initialize, 3 );
  rb_define_singleton_method( RuNeNe_GradientDescent_RMSProp, "from_h", gd_rmsprop_rbclass__from_h, 1 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "initialize_copy", gd_rmsprop_rbobject__initialize_copy, 1 );

  // GradientDescent_RMSProp attributes
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "num_params", gd_rmsprop_rbobject__get_num_params, 0 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "decay", gd_rmsprop_rbobject__get_decay, 0 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "decay=", gd_rmsprop_rbobject__set_decay, 1 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "epsilon", gd_rmsprop_rbobject__get_epsilon, 0 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "epsilon=", gd_rmsprop_rbobject__set_epsilon, 1 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "av_squared_grads", gd_rmsprop_rbobject__get_narr_av_squared_grads, 0 );

  // GradientDescent_RMSProp instance methods
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "pre_gradient_step", gd_rmsprop_rbobject__pre_gradient_step, 2 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "gradient_step", gd_rmsprop_rbobject__gradient_step, 3 );
}
