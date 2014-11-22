// ext/ru_ne_ne/ru_ne_ne.c

#include "ruby_module_ru_ne_ne.h"

/*
 *  Naming conventions used in this C code:
 *
 *  File names
 *    ruby_module_<foo>     :  Ruby bindings for module Foo
 *    ruby_class_<bar>      :  Ruby bindings for class Bar
 *    struct_<baz>          :  C structs for class or module Baz, with memory-management and "methods"
 *    core_<feature>        :  Base C code that works with ints, floats and pointers
 *
 *  Method names
 *    core_<description>    :  Base C code with little or no Ruby interaction, and no validations
 *    p_layer_ff_<desc>    :  OO-style code that takes an s_Layer_FF C struct as first param
 *    ruby_layer_ff_<desc> :  Ruby-bound method for working with RuNeNe::Layer::FeedForward object
 *
*/

void Init_ru_ne_ne() {
  init_module_ru_ne_ne();
}