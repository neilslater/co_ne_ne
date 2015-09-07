// ext/ru_ne_ne/struct_gd_nag.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definition for GradientDescent_NAG and declarations for its memory management
//

#ifndef STRUCT_GD_NAG_H
#define STRUCT_GD_NAG_H

#include <ruby.h>
#include "narray.h"

typedef struct _gd_nag_raw {
  int num_params;
  float momentum;
  VALUE narr_param_update_velocity;
  float *param_update_velocity;
  } GradientDescent_NAG;

GradientDescent_NAG *gd_nag__create();

void gd_nag__init( GradientDescent_NAG *gd_nag, VALUE params, float momentum );

void gd_nag__destroy( GradientDescent_NAG *gd_nag );

void gd_nag__gc_mark( GradientDescent_NAG *gd_nag );

void gd_nag__deep_copy( GradientDescent_NAG *gd_nag_copy, GradientDescent_NAG *gd_nag_orig );

GradientDescent_NAG * gd_nag__clone( GradientDescent_NAG *gd_nag_orig );

void gd_nag__pre_gradient_step( GradientDescent_NAG *gd_nag, float *params, float lr );

void gd_nag__gradient_step( GradientDescent_NAG *gd_nag, float *params, float *gradients, float lr );

#endif
