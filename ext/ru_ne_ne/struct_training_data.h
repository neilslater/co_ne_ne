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
#include "core_shuffle.h"

typedef struct _training_data_raw {
    int input_item_size;
    int output_item_size;
    int input_item_rank;
    int output_item_rank;
    int *input_item_shape;
    int *output_item_shape;
    int *pos_idx;
    int current_pos;
    int num_items;
    VALUE narr_inputs;
    VALUE narr_outputs;
  } TrainingData;

TrainingData *training_data__create();

void training_data__init( TrainingData *training_data, int input_rank, int *input_shape,
      int output_rank, int *output_shape, int num_items );

float *training_data__current_input( TrainingData *training_data );

float *training_data__current_output( TrainingData *training_data );

void training_data__next( TrainingData *training_data );

void training_data__init_from_narray( TrainingData *training_data, VALUE inputs, VALUE outputs );

void training_data__destroy( TrainingData *training_data );

void training_data__gc_mark( TrainingData *training_data );

#endif
