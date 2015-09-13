// ext/ru_ne_ne/struct_gd_rmsprop.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definition for GradientDescent_RMSProp and declarations for its memory management
//

#ifndef STRUCT_GD_RMSPROP_H
#define STRUCT_GD_RMSPROP_H

#include <ruby.h>
#include "narray.h"

typedef struct _gd_rmsprop_raw {
  int num_params;
  float decay;
  float epsilon;
  volatile VALUE narr_av_squared_grads;
  float *av_squared_grads;
  } GradientDescent_RMSProp;

GradientDescent_RMSProp *gd_rmsprop__create();

void gd_rmsprop__init( GradientDescent_RMSProp *gd_rmsprop, VALUE example_params, float decay, float epsilon );

void gd_rmsprop__destroy( GradientDescent_RMSProp *gd_rmsprop );

void gd_rmsprop__gc_mark( GradientDescent_RMSProp *gd_rmsprop );

void gd_rmsprop__deep_copy( GradientDescent_RMSProp *gd_rmsprop_copy, GradientDescent_RMSProp *gd_rmsprop_orig );

GradientDescent_RMSProp * gd_rmsprop__clone( GradientDescent_RMSProp *gd_rmsprop_orig );

void gd_rmsprop__pre_gradient_step( GradientDescent_RMSProp *gd_rmsprop, float *params, float lr );

void gd_rmsprop__gradient_step( GradientDescent_RMSProp *gd_rmsprop, float *params, float *gradients, float lr );

#endif
