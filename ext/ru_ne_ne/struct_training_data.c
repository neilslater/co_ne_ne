// ext/ru_ne_ne/struct_training_data.c

#include "struct_training_data.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating TrainingData structs
//

TrainingData *training_data__create() {
  TrainingData *training_data;
  training_data = xmalloc( sizeof(TrainingData) );
  training_data->narr_inputs = Qnil;
  training_data->narr_outputs = Qnil;
  training_data->inputs = NULL;
  training_data->outputs = NULL;
  training_data->input_item_size = 0;
  training_data->output_item_size = 0;
  training_data->input_item_rank = 0;
  training_data->output_item_rank = 0;
  training_data->input_item_shape = NULL;
  training_data->output_item_shape = NULL;
  training_data->pos_idx = NULL;
  training_data->current_pos = 0;
  training_data->num_items = 0;
  return training_data;
}

void training_data__init( TrainingData *training_data, int input_rank, int *input_shape,
      int output_rank, int *output_shape, int num_items ) {
  int i, size, *pos;
  struct NARRAY *narr;

  training_data->input_item_shape = ALLOC_N( int, input_rank + 1);
  size = 1;
  for( i = 0; i < input_rank; i++ ) {
    training_data->input_item_shape[i] = input_shape[i];
    size *= input_shape[i];
  }
  training_data->input_item_shape[input_rank] = num_items;
  training_data->narr_inputs = na_make_object( NA_SFLOAT, input_rank + 1, training_data->input_item_shape, cNArray );
  training_data->input_item_size = size;
  training_data->input_item_rank = input_rank;
  GetNArray( training_data->narr_inputs, narr );
  training_data->inputs = (float*) narr->ptr;
  na_sfloat_set( narr->total, training_data->inputs, (float) 0.0 );

  training_data->output_item_shape = ALLOC_N( int, output_rank + 1);
  size = 1;
  for( i = 0; i < output_rank; i++ ) {
    training_data->output_item_shape[i] = output_shape[i];
    size *= output_shape[i];
  }
  training_data->output_item_shape[output_rank] = num_items;
  training_data->narr_outputs = na_make_object( NA_SFLOAT, output_rank + 1, training_data->output_item_shape, cNArray );
  training_data->output_item_size = size;
  training_data->output_item_rank = output_rank;
  GetNArray( training_data->narr_outputs, narr );
  training_data->outputs = (float*) narr->ptr;
  na_sfloat_set( narr->total, training_data->outputs, (float) 0.0 );

  training_data->num_items = num_items;

  pos = ALLOC_N( int, num_items );
  for( i = 0; i < num_items; i++ ) {
    pos[i] = i;
  }
  training_data->pos_idx = pos;
  training_data->current_pos = num_items - 1;

  return;
}

float *training_data__current_input( TrainingData *training_data ) {
  return training_data->inputs + training_data->input_item_size * training_data->pos_idx[ training_data->current_pos ];
}

float *training_data__current_output( TrainingData *training_data ) {
  return training_data->outputs + training_data->output_item_size * training_data->pos_idx[ training_data->current_pos ];
}

void training_data__next( TrainingData *training_data ) {
  training_data->current_pos = ( training_data->current_pos + 1 ) % training_data->num_items;

  // Shuffle sequence if we ran out last time
  if ( training_data->current_pos == 0 ) {
    shuffle_ints( training_data->num_items, training_data->pos_idx );
  }

  return;
}

void training_data__destroy( TrainingData *training_data ) {
  xfree( training_data->pos_idx );
  xfree( training_data->input_item_shape );
  xfree( training_data->output_item_shape );
  xfree( training_data );

  // No need to free NArrays - they will be handled by Ruby's GC, and may still be reachable
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void training_data__gc_mark( TrainingData *training_data ) {
  rb_gc_mark( training_data->narr_inputs );
  rb_gc_mark( training_data->narr_outputs );
  return;
}

void training_data__init_from_narray( TrainingData *training_data, VALUE inputs, VALUE outputs ) {
  int *tmp_shape, i, size, *pos, num_items;
  struct NARRAY *na_inputs;
  struct NARRAY *na_outputs;

  training_data->narr_inputs = inputs;
  training_data->narr_outputs = outputs;
  GetNArray( training_data->narr_inputs, na_inputs );
  GetNArray( training_data->narr_outputs, na_outputs );
  training_data->inputs = (float*) na_inputs->ptr;
  training_data->outputs = (float*) na_outputs->ptr;

  training_data->input_item_rank = na_inputs->rank - 1;
  training_data->input_item_shape = ALLOC_N( int, na_inputs->rank );
  tmp_shape = na_inputs->shape;
  size = 1;
  for( i = 0; i < na_inputs->rank - 1; i++ ) {
    training_data->input_item_shape[i] = tmp_shape[i];
    size *= tmp_shape[i];
  }
  num_items = tmp_shape[ na_inputs->rank - 1 ];
  training_data->input_item_size = size;

  training_data->output_item_rank = na_outputs->rank - 1;
  training_data->output_item_shape = ALLOC_N( int, na_outputs->rank );
  tmp_shape = na_outputs->shape;
  size = 1;
  for( i = 0; i < na_outputs->rank - 1; i++ ) {
    training_data->output_item_shape[i] = tmp_shape[i];
    size *= tmp_shape[i];
  }
  training_data->output_item_size = size;

  pos = ALLOC_N( int, num_items );
  for( i = 0; i < num_items; i++ ) {
    pos[i] = i;
  }
  training_data->pos_idx = pos;
  training_data->current_pos = num_items - 1;
  training_data->num_items = num_items;
  return;
}

// Call this after clone or similar operation
void training_data__reinit( TrainingData *training_data ) {
  int *tmp_shape, i, size, *pos, num_items;
  struct NARRAY *na_inputs;
  struct NARRAY *na_outputs;

  xfree( training_data->pos_idx );
  xfree( training_data->input_item_shape );
  xfree( training_data->output_item_shape );

  GetNArray( training_data->narr_inputs, na_inputs );
  GetNArray( training_data->narr_outputs, na_outputs );
  training_data->inputs = (float*) na_inputs->ptr;
  training_data->outputs = (float*) na_outputs->ptr;

  training_data->input_item_rank = na_inputs->rank - 1;
  training_data->input_item_shape = ALLOC_N( int, na_inputs->rank );
  tmp_shape = na_inputs->shape;
  size = 1;
  for( i = 0; i < na_inputs->rank - 1; i++ ) {
    training_data->input_item_shape[i] = tmp_shape[i];
    size *= tmp_shape[i];
  }
  num_items = tmp_shape[ na_inputs->rank - 1 ];
  training_data->input_item_size = size;

  training_data->output_item_rank = na_outputs->rank - 1;
  training_data->output_item_shape = ALLOC_N( int, na_outputs->rank );
  tmp_shape = na_outputs->shape;
  size = 1;
  for( i = 0; i < na_outputs->rank - 1; i++ ) {
    training_data->output_item_shape[i] = tmp_shape[i];
    size *= tmp_shape[i];
  }
  training_data->output_item_size = size;

  pos = ALLOC_N( int, num_items );
  for( i = 0; i < num_items; i++ ) {
    pos[i] = i;
  }
  training_data->pos_idx = pos;
  training_data->current_pos = num_items - 1;
  training_data->num_items = num_items;
  return;
}
