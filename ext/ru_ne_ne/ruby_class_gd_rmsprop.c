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

/* Document-class: RuNeNe::GradientDescent::RMSProp
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Network method definitions
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

  volatile VALUE example_params;
  struct NARRAY *na_params;
  example_params = na_cast_object( rv_params, NA_SFLOAT );
  GetNArray( example_params, na_params );

  gd_rmsprop__init( gd_rmsprop, na_params->total, NUM2FLT( rv_decay ), NUM2FLT( rv_epsilon ) );

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

/* @!attribute [r] squared_de_dw
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE gd_rmsprop_rbobject__get_narr_squared_de_dw( VALUE self ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );
  return gd_rmsprop->narr_squared_de_dw;
}

/* @!attribute [r] average_squared_de_dw
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE gd_rmsprop_rbobject__get_narr_average_squared_de_dw( VALUE self ) {
  GradientDescent_RMSProp *gd_rmsprop = get_gd_rmsprop_struct( self );
  return gd_rmsprop->narr_average_squared_de_dw;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_gd_rmsprop_class( ) {
  // GradientDescent_RMSProp instantiation and class methods
  rb_define_alloc_func( RuNeNe_GradientDescent_RMSProp, gd_rmsprop_alloc );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "initialize", gd_rmsprop_rbobject__initialize, 3 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "initialize_copy", gd_rmsprop_rbobject__initialize_copy, 1 );

  // GradientDescent_RMSProp attributes
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "num_params", gd_rmsprop_rbobject__get_num_params, 0 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "decay", gd_rmsprop_rbobject__get_decay, 0 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "decay=", gd_rmsprop_rbobject__set_decay, 1 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "epsilon", gd_rmsprop_rbobject__get_epsilon, 0 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "epsilon=", gd_rmsprop_rbobject__set_epsilon, 1 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "squared_de_dw", gd_rmsprop_rbobject__get_narr_squared_de_dw, 0 );
  rb_define_method( RuNeNe_GradientDescent_RMSProp, "average_squared_de_dw", gd_rmsprop_rbobject__get_narr_average_squared_de_dw, 0 );
}
