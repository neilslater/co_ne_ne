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

NetTraining *p_net_training_create();

void p_net_training_init( NetTraining *net_training, int input_rank, int *input_shape,
      int output_rank, int *output_shape, int num_items );

float *p_net_training_current_input( NetTraining *net_training );

float *p_net_training_current_output( NetTraining *net_training );

void p_net_training_next( NetTraining *net_training );

void p_net_training_init_simple( NetTraining *net_training, int input_size,
      int output_size, int num_items );

void p_net_training_init_from_narray( NetTraining *net_training, VALUE inputs, VALUE outputs );

void p_net_training_destroy( NetTraining *net_training );

void p_net_training_gc_mark( NetTraining *net_training );

#endif
