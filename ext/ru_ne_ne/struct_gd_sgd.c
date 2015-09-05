// ext/ru_ne_ne/struct_gd_sgd.c

#include "struct_gd_sgd.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for GradientDescent_SGD memory management
//

GradientDescent_SGD *gd_sgd__create() {
  GradientDescent_SGD *gd_sgd;
  gd_sgd = xmalloc( sizeof(GradientDescent_SGD) );
  gd_sgd->num_params = 0;
  return gd_sgd;
}

void gd_sgd__destroy( GradientDescent_SGD *gd_sgd ) {
  xfree( gd_sgd );
  return;
}

void gd_sgd__gc_mark( GradientDescent_SGD *gd_sgd ) {
  return;
}

void gd_sgd__deep_copy( GradientDescent_SGD *gd_sgd_copy, GradientDescent_SGD *gd_sgd_orig ) {
  gd_sgd_copy->num_params = gd_sgd_orig->num_params;

  return;
}

GradientDescent_SGD * gd_sgd__clone( GradientDescent_SGD *gd_sgd_orig ) {
  GradientDescent_SGD * gd_sgd_copy = gd_sgd__create();
  gd_sgd__deep_copy( gd_sgd_copy, gd_sgd_orig );
  return gd_sgd_copy;
}
