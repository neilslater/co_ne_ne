// ext/ru_ne_ne/struct_mbgd.c

#include "struct_mbgd.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Definitions for MBGD memory management
//

MBGD *mbgd__create() {
  MBGD *mbgd;
  mbgd = xmalloc( sizeof(MBGD) );
  mbgd->mbgd_layers = NULL;
  mbgd->num_layers = 0;
  mbgd->num_inputs = 0;
  mbgd->num_outputs = 0;
  return mbgd;
}

void mbgd__init( MBGD *mbgd, int num_mbgd_layers, VALUE *mbgd_layers ) {
  int i, last_num_outputs;
  MBGDLayer *mbgd_layer;

  mbgd->num_layers = num_mbgd_layers;
  mbgd->mbgd_layers = ALLOC_N( VALUE, num_mbgd_layers );

  for ( i = 0; i < mbgd->num_layers; i++ ) {
    Data_Get_Struct( mbgd_layers[i], MBGDLayer, mbgd_layer );
    if ( i == 0 ) {
      mbgd->num_inputs = mbgd_layer->num_inputs;
    } else {
      if ( mbgd_layer->num_inputs != last_num_outputs ) {
        rb_raise( rb_eRuntimeError, "When building mbgd, layer connections failed between output size %d and next input size %d",
            last_num_outputs, mbgd_layer->num_inputs );
      }
    }
    last_num_outputs = mbgd_layer->num_outputs;

    mbgd->mbgd_layers[i] = mbgd_layers[i];
  }

  mbgd->num_outputs = last_num_outputs;

  return;
}


void mbgd__destroy( MBGD *mbgd ) {
  xfree( mbgd->mbgd_layers );
  xfree( mbgd );
  return;
}

void mbgd__gc_mark( MBGD *mbgd ) {
  int i;
  for ( i = 0; i < mbgd->num_layers; i++ ) {
    rb_gc_mark( mbgd->mbgd_layers[i] );
  }
  return;
}

void mbgd__deep_copy( MBGD *mbgd_copy, MBGD *mbgd_orig ) {
  mbgd_copy->num_layers = mbgd_orig->num_layers;
  mbgd_copy->num_inputs = mbgd_orig->num_inputs;
  mbgd_copy->num_outputs = mbgd_orig->num_outputs;

  mbgd_copy->mbgd_layers = ALLOC_N( VALUE, mbgd_copy->num_layers );
  int i;
  for ( i = 0; i < mbgd_copy->num_layers; i++ ) {
    // This calls .clone of each layer via Ruby
    mbgd_copy->mbgd_layers[i] = rb_funcall( mbgd_orig->mbgd_layers[i], rb_intern("clone"), 0 );
  }

  return;
}

MBGD * mbgd__clone( MBGD *mbgd_orig ) {
  MBGD * mbgd_copy = mbgd__create();
  mbgd__deep_copy( mbgd_copy, mbgd_orig );
  return mbgd_copy;
}
