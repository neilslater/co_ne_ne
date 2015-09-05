// ext/ru_ne_ne/ruby_class_gd_nag.c

#include "ruby_class_gd_nag.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_gd_nag.c
//

inline VALUE gd_nag_as_ruby_class( GradientDescent_NAG *gd_nag , VALUE klass ) {
  return Data_Wrap_Struct( klass, gd_nag__gc_mark, gd_nag__destroy, gd_nag );
}

VALUE gd_nag_alloc(VALUE klass) {
  return gd_nag_as_ruby_class( gd_nag__create(), klass );
}

inline GradientDescent_NAG *get_gd_nag_struct( VALUE obj ) {
  GradientDescent_NAG *gd_nag;
  Data_Get_Struct( obj, GradientDescent_NAG, gd_nag );
  return gd_nag;
}

void assert_value_wraps_gd_nag( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)gd_nag__destroy) {
    rb_raise( rb_eTypeError, "Expected a GradientDescent_NAG object, but got something else" );
  }
}

/* Document-class: RuNeNe::GradientDescent::NAG
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Network method definitions
//

/* @overload initialize( num_params, momentum )
 * Creates a new ...
 * @param [Integer] num_params ...
 * @param [Float] momentum ...
 * @return [RuNeNe::GradientDescent::NAG] new ...
 */
VALUE gd_nag_rbobject__initialize( VALUE self, VALUE rv_num_params, VALUE rv_momentum ) {
  GradientDescent_NAG *gd_nag = get_gd_nag_struct( self );

  gd_nag__init( gd_nag, NUM2INT( rv_num_params ), NUM2FLT( rv_momentum ) );

  return self;
}

/* @overload clone
 * When cloned, the returned GradientDescent_NAG has deep copies of C data.
 * @return [RuNeNe::GradientDescent::NAG] new
 */
VALUE gd_nag_rbobject__initialize_copy( VALUE copy, VALUE orig ) {
  GradientDescent_NAG *gd_nag_copy;
  GradientDescent_NAG *gd_nag_orig;

  if (copy == orig) return copy;
  gd_nag_orig = get_gd_nag_struct( orig );
  gd_nag_copy = get_gd_nag_struct( copy );

  gd_nag__deep_copy( gd_nag_copy, gd_nag_orig );

  return copy;
}

/* @!attribute [r] num_params
 * Description goes here
 * @return [Integer]
 */
VALUE gd_nag_rbobject__get_num_params( VALUE self ) {
  GradientDescent_NAG *gd_nag = get_gd_nag_struct( self );
  return INT2NUM( gd_nag->num_params );
}

/* @!attribute momentum
 * Description goes here
 * @return [Float]
 */
VALUE gd_nag_rbobject__get_momentum( VALUE self ) {
  GradientDescent_NAG *gd_nag = get_gd_nag_struct( self );
  return FLT2NUM( gd_nag->momentum );
}

VALUE gd_nag_rbobject__set_momentum( VALUE self, VALUE rv_momentum ) {
  GradientDescent_NAG *gd_nag = get_gd_nag_struct( self );
  gd_nag->momentum = NUM2FLT( rv_momentum );
  return rv_momentum;
}

/* @!attribute [r] weight_velocity
 * Description goes here
 * @return [NArray<sfloat>]
 */
VALUE gd_nag_rbobject__get_narr_weight_velocity( VALUE self ) {
  GradientDescent_NAG *gd_nag = get_gd_nag_struct( self );
  return gd_nag->narr_weight_velocity;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_gd_nag_class( ) {
  // GradientDescent_NAG instantiation and class methods
  rb_define_alloc_func( RuNeNe_GradientDescent_NAG, gd_nag_alloc );
  rb_define_method( RuNeNe_GradientDescent_NAG, "initialize", gd_nag_rbobject__initialize, 2 );
  rb_define_method( RuNeNe_GradientDescent_NAG, "initialize_copy", gd_nag_rbobject__initialize_copy, 1 );

  // GradientDescent_NAG attributes
  rb_define_method( RuNeNe_GradientDescent_NAG, "num_params", gd_nag_rbobject__get_num_params, 0 );
  rb_define_method( RuNeNe_GradientDescent_NAG, "momentum", gd_nag_rbobject__get_momentum, 0 );
  rb_define_method( RuNeNe_GradientDescent_NAG, "momentum=", gd_nag_rbobject__set_momentum, 1 );
  rb_define_method( RuNeNe_GradientDescent_NAG, "weight_velocity", gd_nag_rbobject__get_narr_weight_velocity, 0 );
}
