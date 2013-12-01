// ext/con_ne_ne/struct_net_training.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating TrainingSet structs
//

#ifndef STRUCT_NET_TRAINING_H
#define STRUCT_NET_TRAINING_H

#include <ruby.h>
#include "narray.h"
#include "core_narray.h"

typedef struct _net_training_raw {
    int random_sequence;
    int input_item_size;
    int output_item_size;
    int *pos_idx;
    int current_pos;
    int num_items;
    VALUE narr_inputs;
    VALUE narr_outputs;
  } NetTraining;

#endif
