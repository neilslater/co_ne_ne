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

/* Document-class: RuNeNe::Learn::MBGD
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Network method definitions
//

/* @overload initialize( mbgd_layers )
 * Creates a new ...
 * @param [Object] mbgd_layers ...
 * @return [RuNeNe::Learn::MBGD] new ...
 */
VALUE mbgd_rbobject__initialize( VALUE self, VALUE rv_mbgd_layers ) {
  MBGD *mbgd = get_mbgd_struct( self );

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

/* @!attribute [r] mbgd_layers
 * Description goes here
 * @return [Object]
 */
VALUE mbgd_rbobject__get_mbgd_layers( VALUE self ) {
  MBGD *mbgd = get_mbgd_struct( self );
  return mbgd->mbgd_layers;
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

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mbgd_class( ) {
  // MBGD instantiation and class methods
  rb_define_alloc_func( RuNeNe_Learn_MBGD, mbgd_alloc );
  rb_define_method( RuNeNe_Learn_MBGD, "initialize", mbgd_rbobject__initialize, 1 );
  rb_define_method( RuNeNe_Learn_MBGD, "initialize_copy", mbgd_rbobject__initialize_copy, 1 );

  // MBGD attributes
  rb_define_method( RuNeNe_Learn_MBGD, "mbgd_layers", mbgd_rbobject__get_mbgd_layers, 0 );
  rb_define_method( RuNeNe_Learn_MBGD, "num_layers", mbgd_rbobject__get_num_layers, 0 );
  rb_define_method( RuNeNe_Learn_MBGD, "num_inputs", mbgd_rbobject__get_num_inputs, 0 );
  rb_define_method( RuNeNe_Learn_MBGD, "num_outputs", mbgd_rbobject__get_num_outputs, 0 );
}
