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
 * @param [Array<RuNeNe::Learn::MBGD::Layer>] mbgd_layers ...
 * @return [RuNeNe::Learn::MBGD] new ...
 */
VALUE mbgd_rbobject__initialize( VALUE self, VALUE rv_mbgd_layers ) {
  MBGD *mbgd = get_mbgd_struct( self );

  // This stack-based var avoids memory leaks from alloc which might not be freed on error
  VALUE layers[100];
  volatile VALUE current_layer;
  int i, n;

  Check_Type( rv_mbgd_layers, T_ARRAY );

  n = FIX2INT( rb_funcall( rv_mbgd_layers, rb_intern("count"), 0 ) );
  if ( n < 1 ) {
    rb_raise( rb_eArgError, "no layers in mbgd" );
  }
  if ( n > 100 ) {
    rb_raise( rb_eArgError, "too many layers in mbgd" );
  }

  for ( i = 0; i < n; i++ ) {
    current_layer = rb_ary_entry( rv_mbgd_layers, i );
    // TODO: Accept more than one definition of layer (e.g. orig object, hash). Support
    //       multiple layer types in theory.
    assert_value_wraps_mbgd_layer( current_layer );
    layers[i] = current_layer;
  }

  mbgd__init( mbgd, n, layers);

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

  int i;

  volatile VALUE rv_layers = rb_ary_new2( mbgd->num_layers );
  for ( i = 0; i < mbgd->num_layers; i++ ) {
    rb_ary_store( rv_layers, i, mbgd->mbgd_layers[i] );
  }

  return rv_layers;
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
