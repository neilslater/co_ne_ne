// ext/co_ne_ne/transfer_module.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of Transfer module
//

#ifndef TRANSFER_MODULE_H
#define TRANSFER_MODULE_H

#define NUM2FLT(x) ((float)NUM2DBL(x))
#define FLT2NUM(x) (rb_float_new((double)x))

void init_transfer_module( VALUE parent_module );

#endif
