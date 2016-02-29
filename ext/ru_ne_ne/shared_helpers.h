// ext/kaggle_skeleton/shared_helpers.h

#ifndef SHARED_HELPERS_H
#define SHARED_HELPERS_H

#include "ruby.h"

#define NUM2FLT(x) ((float)NUM2DBL(x))
#define FLT2NUM(x) (rb_float_new((double)x))

// Force inclusion of hash declarations (only MRI includes by default)
#ifdef HAVE_RUBY_ST_H
#include "ruby/st.h"
#else
#include "st.h"
#endif

// Hash lookup helper
VALUE ValAtSymbol(VALUE hash, const char* key);

#endif
