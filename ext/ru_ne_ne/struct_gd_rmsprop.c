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
  gd_rmsprop->narr_av_squared_grads = Qnil;
  gd_rmsprop->av_squared_grads = NULL;
  return gd_rmsprop;
}

void gd_rmsprop__init( GradientDescent_RMSProp *gd_rmsprop, VALUE example_params, float decay, float epsilon ) {
  int i;
  struct NARRAY *narr;
  float *narr_av_squared_grads_ptr;

  gd_rmsprop->decay = decay;

  gd_rmsprop->epsilon = epsilon;

  gd_rmsprop->narr_av_squared_grads = na_clone( example_params );
  GetNArray( gd_rmsprop->narr_av_squared_grads, narr );
  narr_av_squared_grads_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_av_squared_grads_ptr[i] = 1.0;
  }
  gd_rmsprop->av_squared_grads = (float *) narr->ptr;
  gd_rmsprop->num_params = narr->total;

  return;
}

void gd_rmsprop__destroy( GradientDescent_RMSProp *gd_rmsprop ) {
  xfree( gd_rmsprop );
  return;
}

void gd_rmsprop__gc_mark( GradientDescent_RMSProp *gd_rmsprop ) {
  rb_gc_mark( gd_rmsprop->narr_av_squared_grads );
  return;
}

void gd_rmsprop__deep_copy( GradientDescent_RMSProp *gd_rmsprop_copy, GradientDescent_RMSProp *gd_rmsprop_orig ) {
  struct NARRAY *narr;

  gd_rmsprop_copy->num_params = gd_rmsprop_orig->num_params;
  gd_rmsprop_copy->decay = gd_rmsprop_orig->decay;
  gd_rmsprop_copy->epsilon = gd_rmsprop_orig->epsilon;

  gd_rmsprop_copy->narr_av_squared_grads = na_clone( gd_rmsprop_orig->narr_av_squared_grads );
  GetNArray( gd_rmsprop_copy->narr_av_squared_grads, narr );
  gd_rmsprop_copy->av_squared_grads = (float *) narr->ptr;

  return;
}

GradientDescent_RMSProp * gd_rmsprop__clone( GradientDescent_RMSProp *gd_rmsprop_orig ) {
  GradientDescent_RMSProp * gd_rmsprop_copy = gd_rmsprop__create();
  gd_rmsprop__deep_copy( gd_rmsprop_copy, gd_rmsprop_orig );
  return gd_rmsprop_copy;
}

void gd_rmsprop__pre_gradient_step( GradientDescent_RMSProp *gd_rmsprop, float *params, float lr ) {
  return;
}

void gd_rmsprop__gradient_step( GradientDescent_RMSProp *gd_rmsprop, float *params, float *gradients, float lr ) {
  int i;
  float u = 1.0 - gd_rmsprop->decay;
  for( i = 0; i < gd_rmsprop->num_params; i++ ) {
    gd_rmsprop->av_squared_grads[i] = ( gd_rmsprop->decay * gd_rmsprop->av_squared_grads[i] ) + u * gradients[i] * gradients[i];
    params[i] -= lr * gradients[i]/( sqrt( gd_rmsprop->av_squared_grads[i] + gd_rmsprop->epsilon ) );
  }
  return;
}
