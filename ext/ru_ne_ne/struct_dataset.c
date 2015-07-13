// ext/ru_ne_ne/struct_dataset.c

#include "struct_dataset.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating DataSet structs
//

DataSet *dataset__create() {
  DataSet *dataset;
  dataset = xmalloc( sizeof(DataSet) );
  dataset->narr_inputs = Qnil;
  dataset->narr_outputs = Qnil;
  dataset->inputs = NULL;
  dataset->outputs = NULL;
  dataset->input_item_size = 0;
  dataset->output_item_size = 0;
  dataset->input_item_rank = 0;
  dataset->output_item_rank = 0;
  dataset->input_item_shape = NULL;
  dataset->output_item_shape = NULL;
  dataset->pos_idx = NULL;
  dataset->current_pos = 0;
  dataset->num_items = 0;
  return dataset;
}

void dataset__init( DataSet *dataset, int input_rank, int *input_shape,
      int output_rank, int *output_shape, int num_items ) {
  int i, size, *pos;
  struct NARRAY *narr;

  dataset->input_item_shape = ALLOC_N( int, input_rank + 1);
  size = 1;
  for( i = 0; i < input_rank; i++ ) {
    dataset->input_item_shape[i] = input_shape[i];
    size *= input_shape[i];
  }
  dataset->input_item_shape[input_rank] = num_items;
  dataset->narr_inputs = na_make_object( NA_SFLOAT, input_rank + 1, dataset->input_item_shape, cNArray );
  dataset->input_item_size = size;
  dataset->input_item_rank = input_rank;
  GetNArray( dataset->narr_inputs, narr );
  dataset->inputs = (float*) narr->ptr;
  na_sfloat_set( narr->total, dataset->inputs, (float) 0.0 );

  dataset->output_item_shape = ALLOC_N( int, output_rank + 1);
  size = 1;
  for( i = 0; i < output_rank; i++ ) {
    dataset->output_item_shape[i] = output_shape[i];
    size *= output_shape[i];
  }
  dataset->output_item_shape[output_rank] = num_items;
  dataset->narr_outputs = na_make_object( NA_SFLOAT, output_rank + 1, dataset->output_item_shape, cNArray );
  dataset->output_item_size = size;
  dataset->output_item_rank = output_rank;
  GetNArray( dataset->narr_outputs, narr );
  dataset->outputs = (float*) narr->ptr;
  na_sfloat_set( narr->total, dataset->outputs, (float) 0.0 );

  dataset->num_items = num_items;

  pos = ALLOC_N( int, num_items );
  for( i = 0; i < num_items; i++ ) {
    pos[i] = i;
  }
  dataset->pos_idx = pos;
  dataset->current_pos = num_items - 1;

  return;
}

float *dataset__current_input( DataSet *dataset ) {
  return dataset->inputs + dataset->input_item_size * dataset->pos_idx[ dataset->current_pos ];
}

float *dataset__current_output( DataSet *dataset ) {
  return dataset->outputs + dataset->output_item_size * dataset->pos_idx[ dataset->current_pos ];
}

void dataset__next( DataSet *dataset ) {
  dataset->current_pos = ( dataset->current_pos + 1 ) % dataset->num_items;

  // Shuffle sequence if we ran out last time
  if ( dataset->current_pos == 0 ) {
    shuffle_ints( dataset->num_items, dataset->pos_idx );
  }

  return;
}

void dataset__destroy( DataSet *dataset ) {
  xfree( dataset->pos_idx );
  xfree( dataset->input_item_shape );
  xfree( dataset->output_item_shape );
  xfree( dataset );

  // No need to free NArrays - they will be handled by Ruby's GC, and may still be reachable
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void dataset__gc_mark( DataSet *dataset ) {
  rb_gc_mark( dataset->narr_inputs );
  rb_gc_mark( dataset->narr_outputs );
  return;
}

void dataset__init_from_narray( DataSet *dataset, VALUE inputs, VALUE outputs ) {
  int *tmp_shape, i, size, *pos, num_items;
  struct NARRAY *na_inputs;
  struct NARRAY *na_outputs;

  dataset->narr_inputs = inputs;
  dataset->narr_outputs = outputs;
  GetNArray( dataset->narr_inputs, na_inputs );
  GetNArray( dataset->narr_outputs, na_outputs );
  dataset->inputs = (float*) na_inputs->ptr;
  dataset->outputs = (float*) na_outputs->ptr;

  dataset->input_item_rank = na_inputs->rank - 1;
  dataset->input_item_shape = ALLOC_N( int, na_inputs->rank );
  tmp_shape = na_inputs->shape;
  size = 1;
  for( i = 0; i < na_inputs->rank - 1; i++ ) {
    dataset->input_item_shape[i] = tmp_shape[i];
    size *= tmp_shape[i];
  }
  num_items = tmp_shape[ na_inputs->rank - 1 ];
  dataset->input_item_size = size;

  dataset->output_item_rank = na_outputs->rank - 1;
  dataset->output_item_shape = ALLOC_N( int, na_outputs->rank );
  tmp_shape = na_outputs->shape;
  size = 1;
  for( i = 0; i < na_outputs->rank - 1; i++ ) {
    dataset->output_item_shape[i] = tmp_shape[i];
    size *= tmp_shape[i];
  }
  dataset->output_item_size = size;

  pos = ALLOC_N( int, num_items );
  for( i = 0; i < num_items; i++ ) {
    pos[i] = i;
  }
  dataset->pos_idx = pos;
  dataset->current_pos = num_items - 1;
  dataset->num_items = num_items;
  return;
}

// Call this after clone or similar operation
void dataset__reinit( DataSet *dataset ) {
  int *tmp_shape, i, size, *pos, num_items;
  struct NARRAY *na_inputs;
  struct NARRAY *na_outputs;

  xfree( dataset->pos_idx );
  xfree( dataset->input_item_shape );
  xfree( dataset->output_item_shape );

  GetNArray( dataset->narr_inputs, na_inputs );
  GetNArray( dataset->narr_outputs, na_outputs );
  dataset->inputs = (float*) na_inputs->ptr;
  dataset->outputs = (float*) na_outputs->ptr;

  dataset->input_item_rank = na_inputs->rank - 1;
  dataset->input_item_shape = ALLOC_N( int, na_inputs->rank );
  tmp_shape = na_inputs->shape;
  size = 1;
  for( i = 0; i < na_inputs->rank - 1; i++ ) {
    dataset->input_item_shape[i] = tmp_shape[i];
    size *= tmp_shape[i];
  }
  num_items = tmp_shape[ na_inputs->rank - 1 ];
  dataset->input_item_size = size;

  dataset->output_item_rank = na_outputs->rank - 1;
  dataset->output_item_shape = ALLOC_N( int, na_outputs->rank );
  tmp_shape = na_outputs->shape;
  size = 1;
  for( i = 0; i < na_outputs->rank - 1; i++ ) {
    dataset->output_item_shape[i] = tmp_shape[i];
    size *= tmp_shape[i];
  }
  dataset->output_item_size = size;

  pos = ALLOC_N( int, num_items );
  for( i = 0; i < num_items; i++ ) {
    pos[i] = i;
  }
  dataset->pos_idx = pos;
  dataset->current_pos = num_items - 1;
  dataset->num_items = num_items;
  return;
}
