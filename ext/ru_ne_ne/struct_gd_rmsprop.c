// ext/ru_ne_ne/struct_gd_rmsprop.c

#include "struct_gd_rmsprop.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for GradientDescent_RMSProp memory management
//

GradientDescent_RMSProp *gd_rmsprop__create() {
  GradientDescent_RMSProp *gd_rmsprop;
  gd_rmsprop = xmalloc( sizeof(GradientDescent_RMSProp) );
  gd_rmsprop->num_params = 0;
  gd_rmsprop->decay = 0.9;
  gd_rmsprop->epsilon = 1.0e-6;
  gd_rmsprop->narr_squared_de_dw = Qnil;
  gd_rmsprop->squared_de_dw = NULL;
  gd_rmsprop->narr_average_squared_de_dw = Qnil;
  gd_rmsprop->average_squared_de_dw = NULL;
  return gd_rmsprop;
}

void gd_rmsprop__init( GradientDescent_RMSProp *gd_rmsprop, int num_params, float decay, float epsilon ) {
  int i;
  struct NARRAY *narr;
  float *narr_squared_de_dw_ptr;
  float *narr_average_squared_de_dw_ptr;
  int *shape = &num_params;

  gd_rmsprop->num_params = num_params;

  gd_rmsprop->decay = decay;

  gd_rmsprop->epsilon = epsilon;

  gd_rmsprop->narr_squared_de_dw = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( gd_rmsprop->narr_squared_de_dw, narr );
  narr_squared_de_dw_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_squared_de_dw_ptr[i] = 0.0;
  }
  gd_rmsprop->squared_de_dw = (float *) narr->ptr;

  gd_rmsprop->narr_average_squared_de_dw = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( gd_rmsprop->narr_average_squared_de_dw, narr );
  narr_average_squared_de_dw_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_average_squared_de_dw_ptr[i] = 0.0;
  }
  gd_rmsprop->average_squared_de_dw = (float *) narr->ptr;

  return;
}

void gd_rmsprop__destroy( GradientDescent_RMSProp *gd_rmsprop ) {
  xfree( gd_rmsprop );
  return;
}

void gd_rmsprop__gc_mark( GradientDescent_RMSProp *gd_rmsprop ) {
  rb_gc_mark( gd_rmsprop->narr_squared_de_dw );
  rb_gc_mark( gd_rmsprop->narr_average_squared_de_dw );
  return;
}

void gd_rmsprop__deep_copy( GradientDescent_RMSProp *gd_rmsprop_copy, GradientDescent_RMSProp *gd_rmsprop_orig ) {
  struct NARRAY *narr;

  gd_rmsprop_copy->num_params = gd_rmsprop_orig->num_params;
  gd_rmsprop_copy->decay = gd_rmsprop_orig->decay;
  gd_rmsprop_copy->epsilon = gd_rmsprop_orig->epsilon;

  gd_rmsprop_copy->narr_squared_de_dw = na_clone( gd_rmsprop_orig->narr_squared_de_dw );
  GetNArray( gd_rmsprop_copy->narr_squared_de_dw, narr );
  gd_rmsprop_copy->squared_de_dw = (float *) narr->ptr;

  gd_rmsprop_copy->narr_average_squared_de_dw = na_clone( gd_rmsprop_orig->narr_average_squared_de_dw );
  GetNArray( gd_rmsprop_copy->narr_average_squared_de_dw, narr );
  gd_rmsprop_copy->average_squared_de_dw = (float *) narr->ptr;

  return;
}

GradientDescent_RMSProp * gd_rmsprop__clone( GradientDescent_RMSProp *gd_rmsprop_orig ) {
  GradientDescent_RMSProp * gd_rmsprop_copy = gd_rmsprop__create();
  gd_rmsprop__deep_copy( gd_rmsprop_copy, gd_rmsprop_orig );
  return gd_rmsprop_copy;
}
