// ext/ru_ne_ne/ru_ne_ne.c

#include "ruby_module_ru_ne_ne.h"

/*
 *  Naming conventions used in this C code:
 *
 *  File names
 *    ruby_module_<foo>     :  Ruby bindings for module
 *    ruby_class_<bar>      :  Ruby bindings for class Bar
 *    struct_<baz>          :  C structs for Baz, with memory-management and OO-style "methods"
 *    core_<feature>        :  Base C code that works with ints, floats etc (*no* Ruby VALUEs)
 *
 *  Variable names
 *    Module_Class_TheThing :  VALUE container for Ruby Class or Module
 *    The_Thing             :  struct type
 *    the_thing             :  pointer to a struct type
 *
 *  Method names
 *    layer_ff__<desc>        :  OO-style code that takes a Layer_FF C struct as first param
 *    layer_ff_object_<desc>  :  Ruby-bound method for RuNeNe::Layer::FeedForward object
 *    layer_ff_class_<desc>   :  Ruby-bound method for RuNeNe::Layer::FeedForward class
 *
*/

void Init_ru_ne_ne() {
  init_module_ru_ne_ne();
}
