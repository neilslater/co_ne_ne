// ext/co_ne_ne/mlp_classes.c

#include <ruby.h>
#include "narray.h"
#include <stdio.h>
#include <xmmintrin.h>

#include "narray_shared.h"
#include "mlp_classes.h"

VALUE MLP = Qnil;
VALUE Layer = Qnil;
VALUE Network = Qnil;

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mlp_classes( VALUE parent_module ) {
  MLP = rb_define_module_under( parent_module, "MLP" );

  Layer = rb_define_class_under( MLP, "Layer", rb_cObject );

  Network = rb_define_class_under( MLP, "Network", rb_cObject );
}
