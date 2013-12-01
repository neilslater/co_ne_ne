// ext/co_ne_ne/struct_training_set.c

#include "struct_training_set.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions of OO-style functions for manipulating TrainingSet structs
//

TrainingSet *p_training_set_create() {
  TrainingSet *training_set;
  training_set = xmalloc( sizeof(TrainingSet) );
  training_set->narr_inputs = Qnil;
  training_set->narr_outputs = Qnil;
  return training_set;
}

// Creates weights, outputs etc
void p_training_set_new_narrays( TrainingSet *training_set, int size ) {
  int shape[2];
  struct NARRAY *narr;
  shape[0] = size;
  training_set->narr_inputs = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( training_set->narr_inputs, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  training_set->narr_outputs = na_make_object( NA_SFLOAT, 1, shape, cNArray );
  GetNArray( training_set->narr_outputs, narr );
  na_sfloat_set( narr->total, (float*) narr->ptr, (float) 0.0 );

  return;
}

void p_training_set_destroy( TrainingSet *training_set ) {
  xfree( training_set );
  // No need to free NArrays - they will be handled by Ruby's GC, and may still be reachable
  return;
}

// Called by Ruby's GC, we have to mark all child objects that could be in Ruby
// space, so that they don't get deleted.
void p_training_set_gc_mark( TrainingSet *training_set ) {
  rb_gc_mark( training_set->narr_inputs );
  rb_gc_mark( training_set->narr_outputs );
  return;
}
