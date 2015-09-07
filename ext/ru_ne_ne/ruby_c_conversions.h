// ext/ru_ne_ne/ruby_c_conversions.h

#ifndef RUBY_C_CONVERSIONS_H
#define RUBY_C_CONVERSIONS_H

#include <ruby.h>
#include "narray.h"
#include "shared_vars.h"
#include "core_objective_functions.h"
#include "core_transfer_functions.h"
#include "struct_mbgd_layer.h"

transfer_type symbol_to_transfer_type( VALUE rv_transfer_type );
VALUE transfer_type_to_module( transfer_type t );
VALUE transfer_type_to_symbol( transfer_type t );

objective_type symbol_to_objective_type( VALUE rv_objective_type );
VALUE objective_type_to_module( objective_type o );
VALUE objective_type_to_symbol( objective_type o );

gradient_descent_type symbol_to_gradient_descent_type( VALUE rv_gdaccel_symbol );
VALUE gradient_descent_type_to_symbol( gradient_descent_type g );

#endif
