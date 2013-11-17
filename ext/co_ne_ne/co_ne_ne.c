// ext/co_ne_ne/co_ne_ne.c

#include "ruby_module_co_ne_ne.h"

/*
 *  Naming conventions used in this C code:
 *
 *  File names
 *    ruby_module_<foo>   :   Ruby bindings for module Foo
 *    ruby_class_<bar>    :   Ruby bindings for class Bar
 *    struct_<baz>        :   C structs for class or module Baz, with memory-management and "methods"
 *    core_<feature>      :   Base C code that works with ints, floats and pointers
 *
*/

void Init_co_ne_ne() {
  init_module_co_ne_ne();
}
