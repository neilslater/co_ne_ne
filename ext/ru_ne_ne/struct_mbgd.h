// ext/ru_ne_ne/struct_mbgd.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definition for MBGD and declarations for its memory management
//

#ifndef STRUCT_MBGD_H
#define STRUCT_MBGD_H

#include <ruby.h>
#include "narray.h"
#include "struct_mbgd_layer.h"
#include "struct_nn_model.h"

typedef struct _mbgd_raw {
  VALUE *mbgd_layers;
  int num_layers;
  int num_inputs;
  int num_outputs;
  } MBGD;

MBGD *mbgd__create();

void mbgd__init( MBGD *mbgd, int num_mbgd_layers, VALUE *mbgd_layers );

void mbgd__destroy( MBGD *mbgd );

void mbgd__gc_mark( MBGD *mbgd );

void mbgd__deep_copy( MBGD *mbgd_copy, MBGD *mbgd_orig );

MBGD * mbgd__clone( MBGD *mbgd_orig );

#endif
