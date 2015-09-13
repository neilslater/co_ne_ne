// ext/con_ne_ne/struct_dataset.h

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

typedef struct _dataset_raw {
    int input_item_size;
    int output_item_size;
    int input_item_rank;
    int output_item_rank;
    int *input_item_shape;
    int *output_item_shape;
    int *pos_idx;
    int current_pos;
    int num_items;
    volatile VALUE narr_inputs;
    volatile VALUE narr_outputs;
    float *inputs;
    float *outputs;
  } DataSet;

DataSet *dataset__create();

void dataset__init( DataSet *dataset, int input_rank, int *input_shape,
      int output_rank, int *output_shape, int num_items );

float *dataset__current_input( DataSet *dataset );

float *dataset__current_output( DataSet *dataset );

void dataset__next( DataSet *dataset );

void dataset__init_from_narray( DataSet *dataset, VALUE inputs, VALUE outputs );

void dataset__destroy( DataSet *dataset );

void dataset__gc_mark( DataSet *dataset );

void dataset__reinit( DataSet *dataset );

#endif
