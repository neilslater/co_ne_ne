// ext/ru_ne_ne/ruby_class_gd_nag.c

#include "ruby_class_gd_nag.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for training data arrays - the deeper implementation is in
//  struct_gd_nag.c
//

// Helper for converting hash to C properties
void copy_hash_to_gd_nag_properties( VALUE rv_opts, GradientDescent_NAG *gd_nag ) {
  volatile VALUE rv_var;
  volatile VALUE new_narray;
  struct NARRAY* narr;

  // Start with simple properties

  rv_var = ValAtSymbol(rv_opts,"momentum");
  if ( !NIL_P(rv_var) ) {
    gd_nag->momentum = NUM2FLT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"num_params");
  if ( !NIL_P(rv_var) ) {
    gd_nag->num_params = NUM2INT( rv_var );
  }

  rv_var = ValAtSymbol(rv_opts,"weight_velocity");
  if ( !NIL_P(rv_var) ) {
    new_narray = na_cast_object(rv_var, NA_SFLOAT);
    GetNArray( new_narray, narr );
    gd_nag->narr_weight_velocity = new_narray;
    gd_nag->weight_velocity = (float *) narr->ptr;
    gd_nag->num_params = narr->total;
  }

  return;
}

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

/* @overload initialize( params, momentum )
 * Creates a new ...
 * @param [NArray<sfloat>] params actual or example NArray of params for optimisation
 * @param [Float] momentum ...
 * @return [RuNeNe::GradientDescent::NAG] new ...
 */
VALUE gd_nag_rbobject__initialize( VALUE self, VALUE rv_params, VALUE rv_momentum ) {
  GradientDescent_NAG *gd_nag = get_gd_nag_struct( self );

  volatile VALUE example_params;
  example_params = na_cast_object( rv_params, NA_SFLOAT );

  gd_nag__init( gd_nag, example_params, NUM2FLT( rv_momentum ) );

  return self;
}

/* @overload initialize( h )
 * Creates a new ...
 * keys are h[:weight_velocity], h[:momentum]
 * @return [RuNeNe::GradientDescent::NAG] new ...
 */

VALUE gd_nag_rbclass__from_h( VALUE self, VALUE rv_h ) {
  GradientDescent_NAG *gd_nag;
  Check_Type( rv_h, T_HASH );

  VALUE rv_gd_nag = gd_nag_alloc( RuNeNe_GradientDescent_NAG );
  gd_nag = get_gd_nag_struct( rv_gd_nag );

  copy_hash_to_gd_nag_properties( rv_h, gd_nag );

  return rv_gd_nag;
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
  rb_define_singleton_method( RuNeNe_GradientDescent_NAG, "from_h", gd_nag_rbclass__from_h, 1 );
  rb_define_method( RuNeNe_GradientDescent_NAG, "initialize_copy", gd_nag_rbobject__initialize_copy, 1 );

  // GradientDescent_NAG attributes
  rb_define_method( RuNeNe_GradientDescent_NAG, "num_params", gd_nag_rbobject__get_num_params, 0 );
  rb_define_method( RuNeNe_GradientDescent_NAG, "momentum", gd_nag_rbobject__get_momentum, 0 );
  rb_define_method( RuNeNe_GradientDescent_NAG, "momentum=", gd_nag_rbobject__set_momentum, 1 );
  rb_define_method( RuNeNe_GradientDescent_NAG, "weight_velocity", gd_nag_rbobject__get_narr_weight_velocity, 0 );
}
