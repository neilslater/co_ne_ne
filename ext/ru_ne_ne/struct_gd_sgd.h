// ext/ru_ne_ne/struct_gd_sgd.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definition for GradientDescent_SGD and declarations for its memory management
//

#ifndef STRUCT_GD_SGD_H
#define STRUCT_GD_SGD_H

#include <ruby.h>
#include "narray.h"

typedef struct _gd_sgd_raw {
  int num_params;
  } GradientDescent_SGD;

GradientDescent_SGD *gd_sgd__create();

void gd_sgd__destroy( GradientDescent_SGD *gd_sgd );

void gd_sgd__gc_mark( GradientDescent_SGD *gd_sgd );

void gd_sgd__deep_copy( GradientDescent_SGD *gd_sgd_copy, GradientDescent_SGD *gd_sgd_orig );

GradientDescent_SGD * gd_sgd__clone( GradientDescent_SGD *gd_sgd_orig );

#endif
