// ext/ru_ne_ne/ruby_c_conversions.h

#ifndef RUBY_C_CONVERSIONS_H
#define RUBY_C_CONVERSIONS_H

#include <ruby.h>
#include "narray.h"
#include "shared_vars.h"
#include "core_objective_functions.h"
#include "core_transfer_functions.h"

transfer_type symbol_to_transfer_type( VALUE rv_transfer_type );
objective_type symbol_to_objective_type( VALUE rv_objective_type );

#endif
