// ext/co_ne_ne/struct_net_training.c

#include "struct_net_training.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating NetTraining structs
//

NetTraining *p_net_training_create() {
  NetTraining *net_training;
  net_training = xmalloc( sizeof(NetTraining) );
  net_training->narr_inputs = Qnil;
  net_training->narr_outputs = Qnil;
  net_training->random_sequence = 0;
  net_training->input_item_size = 0;
  net_training->output_item_size = 0;
  net_training->pos_idx = NULL;
  net_training->current_pos = 0;
  net_training->num_items = 0;
  return net_training;
}

void p_net_training_init( NetTraining *net_training, int input_rank, int *input_shape,
      int output_rank, int *output_shape, int num_items ) {
  int *tmp_shape, i, size, *pos;
  struct NARRAY *narr;

  tmp_shape = ALLOC_N( int, input_rank + 1 );
  size = 1;
  for( i = 0; i < input_rank; i++ ) {
    tmp_shape[i] = input_shape[i];
    size *= input_shape[i];
  }
  tmp_shape[input_rank] = num_items;
  net_training->narr_inputs = na_make_object( NA_SFLOAT, input_rank + 1, tmp_shape, cNArray );
  net_training->input_item_size = size;
  GetNArray( net_training->narr_inputs, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );
  xfree( tmp_shape );

  tmp_shape = ALLOC_N( int, output_rank + 1 );
  size = 1;
  for( i = 0; i < output_rank; i++ ) {
    tmp_shape[i] = output_shape[i];
    size *= output_shape[i];
  }
  tmp_shape[output_rank] = num_items;
  net_training->narr_outputs = na_make_object( NA_SFLOAT, output_rank + 1, tmp_shape, cNArray );
  net_training->output_item_size = size;
  GetNArray( net_training->narr_outputs, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );
  xfree( tmp_shape );

  net_training->num_items = num_items;

  pos = ALLOC_N( int, num_items );
  for( i = 0; i < num_items; i++ ) {
    pos[i] = i;
  }
  net_training->pos_idx = pos;
  net_training->current_pos = 0;

  return;
}

float *p_net_training_current_input( NetTraining *net_training ) {
  int i;
  struct NARRAY *narr;
  GetNArray( net_training->narr_inputs, narr );
  i = net_training->pos_idx[ net_training->current_pos ];
  return (float*) narr->ptr[ net_training->input_item_size * i ];
}

float *p_net_training_current_output( NetTraining *net_training ) {
  int i;
  struct NARRAY *narr;
  GetNArray( net_training->narr_outputs, narr );
  i = net_training->pos_idx[ net_training->current_pos ];
  return (float*) narr->ptr[ net_training->output_item_size * i ];
}

// Init assuming 1-dimensional arrays for input and output
void p_net_training_init_simple( NetTraining *net_training, int input_size, int output_size, int num_items ) {
  int in_shape[1], out_shape[1];
  in_shape[0] = input_size;
  out_shape[0] = output_size;

  p_net_training_init( net_training, 1, in_shape, 1, out_shape, num_items );
  return;
}

void p_net_training_destroy( NetTraining *net_training ) {
  xfree( net_training->pos_idx );
  xfree( net_training );

  // No need to free NArrays - they will be handled by Ruby's GC, and may still be reachable
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void p_net_training_gc_mark( NetTraining *net_training ) {
  rb_gc_mark( net_training->narr_inputs );
  rb_gc_mark( net_training->narr_outputs );
  return;
}
