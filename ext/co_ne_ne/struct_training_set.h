// ext/con_ne_ne/struct_training_set.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating TrainingSet structs
//

#ifndef STRUCT_TRAINING_SET_H
#define STRUCT_TRAINING_SET_H

#include <ruby.h>
#include "narray.h"
#include "core_narray.h"

typedef struct _training_set_raw {
    VALUE narr_inputs;
    VALUE narr_outputs;
  } TrainingSet;

#endif
