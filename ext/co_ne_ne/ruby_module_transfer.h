// ext/co_ne_ne/ruby_module_transfer.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of Transfer module
//

#ifndef RUBY_MODULE_TRANSFER_H
#define RUBY_MODULE_TRANSFER_H

#define NUM2FLT(x) ((float)NUM2DBL(x))
#define FLT2NUM(x) (rb_float_new((double)x))

#include <ruby.h>
#include "narray.h"
#include "core_narray.h"
#include "core_transfer_functions.h"

void init_transfer_module();

extern VALUE Transfer;
extern VALUE Sigmoid;
extern VALUE TanH;
extern VALUE ReLU;

#endif
