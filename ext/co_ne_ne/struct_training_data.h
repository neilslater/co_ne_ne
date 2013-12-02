// ext/con_ne_ne/struct_training_data.h

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Declarations of OO-style functions for manipulating TrainingSet structs
//

#ifndef STRUCT_NET_TRAINING_H
#define STRUCT_NET_TRAINING_H

#include <ruby.h>
#include "narray.h"
#include "core_narray.h"

typedef struct _training_data_raw {
    int random_sequence;
    int input_item_size;
    int output_item_size;
    int *pos_idx;
    int current_pos;
    int num_items;
    VALUE narr_inputs;
    VALUE narr_outputs;
  } TrainingData;

TrainingData *p_training_data_create();

void p_training_data_init( TrainingData *training_data, int input_rank, int *input_shape,
      int output_rank, int *output_shape, int num_items );

float *p_training_data_current_input( TrainingData *training_data );

float *p_training_data_current_output( TrainingData *training_data );

void p_training_data_next( TrainingData *training_data );

void p_training_data_init_simple( TrainingData *training_data, int input_size,
      int output_size, int num_items );

void p_training_data_init_from_narray( TrainingData *training_data, VALUE inputs, VALUE outputs );

void p_training_data_destroy( TrainingData *training_data );

void p_training_data_gc_mark( TrainingData *training_data );

#endif
