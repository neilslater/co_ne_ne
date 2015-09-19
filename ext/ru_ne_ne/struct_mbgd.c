// ext/ru_ne_ne/struct_mbgd.c

#include "struct_mbgd.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for MBGD memory management
//

MBGD *mbgd__create() {
  MBGD *mbgd;
  mbgd = xmalloc( sizeof(MBGD) );
  mbgd->mbgd_layers = Qnil;
  mbgd->num_layers = 0;
  mbgd->num_inputs = 0;
  mbgd->num_outputs = 0;
  return mbgd;
}

void mbgd__destroy( MBGD *mbgd ) {
  xfree( mbgd );
  return;
}

void mbgd__gc_mark( MBGD *mbgd ) {
  rb_gc_mark( mbgd->mbgd_layers );
  return;
}

void mbgd__deep_copy( MBGD *mbgd_copy, MBGD *mbgd_orig ) {
  mbgd_copy->mbgd_layers = mbgd_orig->mbgd_layers;
  mbgd_copy->num_layers = mbgd_orig->num_layers;
  mbgd_copy->num_inputs = mbgd_orig->num_inputs;
  mbgd_copy->num_outputs = mbgd_orig->num_outputs;

  return;
}

MBGD * mbgd__clone( MBGD *mbgd_orig ) {
  MBGD * mbgd_copy = mbgd__create();
  mbgd__deep_copy( mbgd_copy, mbgd_orig );
  return mbgd_copy;
}
