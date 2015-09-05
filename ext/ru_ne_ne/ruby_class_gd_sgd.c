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

/* @overload initialize( num_params )
 * Creates a new ...
 * @param [Integer] num_params ...
 * @return [RuNeNe::GradientDescent::SGD] new ...
 */
VALUE gd_sgd_rbobject__initialize( VALUE self, VALUE rv_num_params ) {
  GradientDescent_SGD *gd_sgd = get_gd_sgd_struct( self );

  gd_sgd->num_params = NUM2INT( rv_num_params );

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

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_gd_sgd_class( ) {
  // GradientDescent_SGD instantiation and class methods
  rb_define_alloc_func( RuNeNe_GradientDescent_SGD, gd_sgd_alloc );
  rb_define_method( RuNeNe_GradientDescent_SGD, "initialize", gd_sgd_rbobject__initialize, 1 );
  rb_define_method( RuNeNe_GradientDescent_SGD, "initialize_copy", gd_sgd_rbobject__initialize_copy, 1 );

  // GradientDescent_SGD attributes
  rb_define_method( RuNeNe_GradientDescent_SGD, "num_params", gd_sgd_rbobject__get_num_params, 0 );
}
