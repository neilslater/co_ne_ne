// ext/ru_ne_ne/ruby_class_gd_sgd.c

#include "ruby_class_gd_sgd.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_gd_sgd.c
//

inline VALUE gd_sgd_as_ruby_class( GradientDescent_SGD *gd_sgd , VALUE klass ) {
  return Data_Wrap_Struct( klass, gd_sgd__gc_mark, gd_sgd__destroy, gd_sgd );
}

VALUE gd_sgd_alloc(VALUE klass) {
  return gd_sgd_as_ruby_class( gd_sgd__create(), klass );
}

inline GradientDescent_SGD *get_gd_sgd_struct( VALUE obj ) {
  GradientDescent_SGD *gd_sgd;
  Data_Get_Struct( obj, GradientDescent_SGD, gd_sgd );
  return gd_sgd;
}

void assert_value_wraps_gd_sgd( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)gd_sgd__destroy) {
    rb_raise( rb_eTypeError, "Expected a GradientDescent_SGD object, but got something else" );
  }
}

/* Document-class: RuNeNe::GradientDescent::SGD
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Network method definitions
//

/* @overload initialize( params )
 * Creates a new ...
 * @param [NArray<sfloat>] params actual or example NArray of params for optimisation
 * @return [RuNeNe::GradientDescent::SGD] new ...
 */
VALUE gd_sgd_rbobject__initialize( VALUE self, VALUE rv_params ) {
  GradientDescent_SGD *gd_sgd = get_gd_sgd_struct( self );

  volatile VALUE example_params;
  struct NARRAY *na_params;
  example_params = na_cast_object( rv_params, NA_SFLOAT );
  GetNArray( example_params, na_params );

  gd_sgd->num_params = na_params->total;

  return self;
}

/* @overload clone
 * When cloned, the returned GradientDescent_SGD has deep copies of C data.
 * @return [RuNeNe::GradientDescent::SGD] new
 */
VALUE gd_sgd_rbobject__initialize_copy( VALUE copy, VALUE orig ) {
  GradientDescent_SGD *gd_sgd_copy;
  GradientDescent_SGD *gd_sgd_orig;

  if (copy == orig) return copy;
  gd_sgd_orig = get_gd_sgd_struct( orig );
  gd_sgd_copy = get_gd_sgd_struct( copy );

  gd_sgd__deep_copy( gd_sgd_copy, gd_sgd_orig );

  return copy;
}

/* @!attribute [r] num_params
 * Description goes here
 * @return [Integer]
 */
VALUE gd_sgd_rbobject__get_num_params( VALUE self ) {
  GradientDescent_SGD *gd_sgd = get_gd_sgd_struct( self );
  return INT2NUM( gd_sgd->num_params );
}

/* @overload pre_gradient_step( params, learning_rate )
 * Prepares object for a gradient step. Some optimisers alter params
 * @param [NArray<sfloat>] params array of same size as initial example
 * @param [Float] learning_rate size of
 * @return [NArray<sfloat>] the params array that willbe optimised (may be cast to NArray<sfloat> from supplied params)
 */
VALUE gd_sgd_rbobject__pre_gradient_step( VALUE self, VALUE rv_params, VALUE rv_learning_rate ) {
  GradientDescent_SGD *gd_sgd = get_gd_sgd_struct( self );

  volatile VALUE opt_params;
  struct NARRAY *na_params;
  opt_params = na_cast_object( rv_params, NA_SFLOAT );
  GetNArray( opt_params, na_params );

  if ( gd_sgd->num_params != na_params->total ) {
    rb_raise( rb_eArgError, "Expecting NArray with %d params, but input has %d params", gd_sgd->num_params, na_params->total );
  }

  gd_sgd__pre_gradient_step( gd_sgd, (float *)na_params->ptr, NUM2FLT(rv_learning_rate) );

  return opt_params;
}

/* @overload pre_gradient_step( params, gradients, learning_rate )
 * Prepares object for a gradient step. Some optimisers alter params
 * @param [NArray<sfloat>] params array of same size as initial example
 * @param [NArray<sfloat>] gradients array of same size as initial example
 * @param [Float] learning_rate size of
 * @return [NArray<sfloat>] the params array that willbe optimised (may be cast to NArray<sfloat> from supplied params)
 */
VALUE gd_sgd_rbobject__gradient_step( VALUE self, VALUE rv_params, VALUE rv_gradients, VALUE rv_learning_rate ) {
  GradientDescent_SGD *gd_sgd = get_gd_sgd_struct( self );

  volatile VALUE opt_params;
  struct NARRAY *na_params;
  volatile VALUE gradients;
  struct NARRAY *na_grads;

  opt_params = na_cast_object( rv_params, NA_SFLOAT );
  GetNArray( opt_params, na_params );

  if ( gd_sgd->num_params != na_params->total ) {
    rb_raise( rb_eArgError, "Expecting NArray with %d params, but input has %d params", gd_sgd->num_params, na_params->total );
  }

  gradients = na_cast_object( rv_gradients, NA_SFLOAT );
  GetNArray( gradients, na_grads );

  if ( gd_sgd->num_params != na_grads->total ) {
    rb_raise( rb_eArgError, "Expecting NArray with %d params, but gradient has %d params", gd_sgd->num_params, na_grads->total );
  }

  gd_sgd__gradient_step( gd_sgd, (float *)na_params->ptr, (float *)na_grads->ptr, NUM2FLT(rv_learning_rate) );

  return opt_params;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void init_gd_sgd_class( ) {
  // GradientDescent_SGD instantiation and class methods
  rb_define_alloc_func( RuNeNe_GradientDescent_SGD, gd_sgd_alloc );
  rb_define_method( RuNeNe_GradientDescent_SGD, "initialize", gd_sgd_rbobject__initialize, 1 );
  rb_define_method( RuNeNe_GradientDescent_SGD, "initialize_copy", gd_sgd_rbobject__initialize_copy, 1 );

  // GradientDescent_SGD attributes
  rb_define_method( RuNeNe_GradientDescent_SGD, "num_params", gd_sgd_rbobject__get_num_params, 0 );

  // GradientDescent_SGD instance methods
  rb_define_method( RuNeNe_GradientDescent_SGD, "pre_gradient_step", gd_sgd_rbobject__pre_gradient_step, 2 );
  rb_define_method( RuNeNe_GradientDescent_SGD, "gradient_step", gd_sgd_rbobject__gradient_step, 3 );
}
