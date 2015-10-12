// ext/ru_ne_ne/ruby_class_dataset.h

#ifndef RUBY_CLASS_NET_TRAINING_H
#define RUBY_CLASS_NET_TRAINING_H

#include <ruby.h>
#include "narray.h"
#include "struct_dataset.h"
#include "shared_vars.h"

void init_dataset_class( );
DataSet *safe_get_dataset_struct( VALUE obj );

#endif
