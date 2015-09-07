// ext/ru_ne_ne/struct_gd_nag.c

#include "struct_gd_nag.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for GradientDescent_NAG memory management
//

GradientDescent_NAG *gd_nag__create() {
  GradientDescent_NAG *gd_nag;
  gd_nag = xmalloc( sizeof(GradientDescent_NAG) );
  gd_nag->num_params = 0;
  gd_nag->momentum = 0.01;
  gd_nag->narr_param_update_velocity = Qnil;
  gd_nag->param_update_velocity = NULL;
  return gd_nag;
}

void gd_nag__init( GradientDescent_NAG *gd_nag, VALUE params, float momentum ) {
  int i;
  struct NARRAY *narr;
  float *narr_param_update_velocity_ptr;

  gd_nag->momentum = momentum;

  gd_nag->narr_param_update_velocity = na_clone( params );
  GetNArray( gd_nag->narr_param_update_velocity, narr );
  narr_param_update_velocity_ptr = (float*) narr->ptr;
  for( i = 0; i < narr->total; i++ ) {
    narr_param_update_velocity_ptr[i] = 0.0;
  }
  gd_nag->param_update_velocity = (float *) narr->ptr;
  gd_nag->num_params = narr->total;

  return;
}

void gd_nag__destroy( GradientDescent_NAG *gd_nag ) {
  xfree( gd_nag );
  return;
}

void gd_nag__gc_mark( GradientDescent_NAG *gd_nag ) {
  rb_gc_mark( gd_nag->narr_param_update_velocity );
  return;
}

void gd_nag__deep_copy( GradientDescent_NAG *gd_nag_copy, GradientDescent_NAG *gd_nag_orig ) {
  struct NARRAY *narr;

  gd_nag_copy->num_params = gd_nag_orig->num_params;
  gd_nag_copy->momentum = gd_nag_orig->momentum;

  gd_nag_copy->narr_param_update_velocity = na_clone( gd_nag_orig->narr_param_update_velocity );
  GetNArray( gd_nag_copy->narr_param_update_velocity, narr );
  gd_nag_copy->param_update_velocity = (float *) narr->ptr;

  return;
}

GradientDescent_NAG * gd_nag__clone( GradientDescent_NAG *gd_nag_orig ) {
  GradientDescent_NAG * gd_nag_copy = gd_nag__create();
  gd_nag__deep_copy( gd_nag_copy, gd_nag_orig );
  return gd_nag_copy;
}

void gd_nag__pre_gradient_step( GradientDescent_NAG *gd_nag, float *params, float lr ) {
  // For Nesterov momentum, we take a step here
  int i;

  for( i = 0; i < gd_nag->num_params; i++ ) {
    gd_nag->param_update_velocity[i] *= gd_nag->momentum;
    params[i] += gd_nag->param_update_velocity[i];
  }
  return;
}

void gd_nag__gradient_step( GradientDescent_NAG *gd_nag, float *params, float *gradients, float lr ) {
  int i;
  float u;
  for( i = 0; i < gd_nag->num_params; i++ ) {
    u = lr * gradients[i];
    gd_nag->param_update_velocity[i] -= u;
    params[i] -= u;
  }
  return;
}
