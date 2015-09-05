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
  VALUE narr_squared_de_dw;
  float *squared_de_dw;
  VALUE narr_average_squared_de_dw;
  float *average_squared_de_dw;
  } GradientDescent_RMSProp;

GradientDescent_RMSProp *gd_rmsprop__create();

void gd_rmsprop__init( GradientDescent_RMSProp *gd_rmsprop, int num_params, float decay, float epsilon );

void gd_rmsprop__destroy( GradientDescent_RMSProp *gd_rmsprop );

void gd_rmsprop__gc_mark( GradientDescent_RMSProp *gd_rmsprop );

void gd_rmsprop__deep_copy( GradientDescent_RMSProp *gd_rmsprop_copy, GradientDescent_RMSProp *gd_rmsprop_orig );

GradientDescent_RMSProp * gd_rmsprop__clone( GradientDescent_RMSProp *gd_rmsprop_orig );

#endif
