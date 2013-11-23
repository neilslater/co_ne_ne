// ext/co_ne_ne/ruby_class_mlp_network.c

#include "ruby_class_mlp_network.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Ruby bindings for multi-layer perceptron code - the deeper implementation is in
//  struct_mlp_layer.c and struct_mlp_network.c
//

inline VALUE mlp_network_as_ruby_class( MLP_Network *mlp_network , VALUE klass ) {
  return Data_Wrap_Struct( klass, p_mlp_network_gc_mark, p_mlp_network_destroy, mlp_network );
}

VALUE mlp_network_alloc(VALUE klass) {
  return mlp_network_as_ruby_class( p_mlp_network_create(), klass );
}

inline MLP_Network *get_mlp_network_struct( VALUE obj ) {
  MLP_Network *mlp_network;
  Data_Get_Struct( obj, MLP_Network, mlp_network );
  return mlp_network;
}

void assert_value_wraps_mlp_network( VALUE obj ) {
  if ( TYPE(obj) != T_DATA ||
      RDATA(obj)->dfree != (RUBY_DATA_FUNC)p_mlp_network_destroy) {
    rb_raise( rb_eTypeError, "Expected a Network object, but got something else" );
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Layer method definitions
//

VALUE mlp_network_class_initialize( VALUE self, VALUE num_inputs, VALUE hidden_layers, VALUE num_outputs ) {
  int ninputs, noutputs, i, nhlayers, hlsize, *layer_sizes;
  MLP_Network *mlp_network = get_mlp_network_struct( self );
  ninputs = NUM2INT( num_inputs );
  noutputs = NUM2INT( num_outputs );

  if (ninputs < 1) {
    rb_raise( rb_eArgError, "Input size %d not allowed.", ninputs );
  }

  if (noutputs < 1) {
    rb_raise( rb_eArgError, "Output size %d not allowed.", noutputs );
  }

  // Pre-check all array entries before initialising further
  Check_Type( hidden_layers, T_ARRAY );
  nhlayers = FIX2INT( rb_funcall( hidden_layers, rb_intern("count"), 0 ) );
  for ( i = 0; i < nhlayers; i++ ) {
    hlsize = FIX2INT( rb_ary_entry( hidden_layers, i ) );
    if ( hlsize < 1 ) {
      rb_raise( rb_eArgError, "Hidden layer output size %d not allowed.", hlsize );
    }
  }

  layer_sizes = ALLOC_N( int, nhlayers + 2 );
  layer_sizes[0] = ninputs;
  for ( i = 0; i < nhlayers; i++ ) {
    layer_sizes[i+1] = FIX2INT( rb_ary_entry( hidden_layers, i ) );
  }
  layer_sizes[nhlayers+1] = noutputs;

  p_mlp_network_init_layers( mlp_network, nhlayers + 1, layer_sizes );

  xfree( layer_sizes );
  return self;
}

// Special initialize to support "clone"
VALUE mlp_network_class_initialize_copy( VALUE copy, VALUE orig ) {
  rb_raise( rb_eArgError, "Cannot clone CoNeNe::MLP::Network (yet)." );
  return copy;
}

VALUE mlp_network_object_num_layers( VALUE self ) {
  int count = 0;
  VALUE layer_object;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    count++;
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  return INT2NUM( count );
}

VALUE mlp_network_object_layers( VALUE self ) {
  int count = 0;
  VALUE layer_object, all_layers;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    count++;
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  all_layers = rb_ary_new2( count );
  count = 0;
  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    rb_ary_store( all_layers, count, layer_object );
    count++;
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  return all_layers;
}

VALUE mlp_network_object_init_weights( int argc, VALUE* argv, VALUE self ) {
  VALUE minw, maxw;
  float min_weight, max_weight;
  VALUE layer_object;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  rb_scan_args( argc, argv, "02", &minw, &maxw );

  if ( ! NIL_P(minw) ) {
    min_weight = NUM2FLT( minw );
    if ( ! NIL_P(maxw) ) {
      max_weight = NUM2FLT( maxw );
    } else {
      max_weight = min_weight;
      min_weight = -max_weight;
    }
  } else {
    min_weight = -0.8;
    max_weight = 0.8;
  }

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    p_mlp_layer_init_weights( mlp_layer, min_weight, max_weight );
    layer_object = mlp_layer->output_layer;
  }

  return Qnil;
}

VALUE mlp_network_object_num_outputs( VALUE self ) {
  int count_outputs = 0;
  VALUE layer_object;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    count_outputs = mlp_layer->num_outputs;
    layer_object = mlp_layer->output_layer;
  }

  return INT2NUM( count_outputs );
}

VALUE mlp_network_object_num_inputs( VALUE self ) {
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  Data_Get_Struct( mlp_network->first_layer, MLP_Layer, mlp_layer );
  return INT2NUM( mlp_layer->num_inputs );
}

VALUE mlp_network_object_output( VALUE self ) {
  VALUE layer_object;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  return mlp_layer->narr_output;
}

VALUE mlp_network_object_input( VALUE self ) {
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  Data_Get_Struct( mlp_network->first_layer, MLP_Layer, mlp_layer );
  return mlp_layer->narr_input;
}

VALUE mlp_network_object_run( VALUE self, VALUE new_input ) {
  struct NARRAY *na_input;
  volatile VALUE val_input;
  volatile VALUE layer_object;
  MLP_Layer *mlp_old_input_layer;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  layer_object = mlp_network->first_layer;
  Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );

  val_input = na_cast_object(new_input, NA_SFLOAT);
  GetNArray( val_input, na_input );

  if ( na_input->rank != 1 ) {
    rb_raise( rb_eArgError, "Inputs rank should be 1, but got %d", na_input->rank );
  }

  if ( na_input->total != mlp_layer->num_inputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer input size %d", na_input->total, mlp_layer->num_inputs );
  }

  if ( ! NIL_P( mlp_layer->input_layer ) ) {
    // This layer has an existing input layer, it needs to stop pointing its output here
    Data_Get_Struct( mlp_layer->input_layer, MLP_Layer, mlp_old_input_layer );
    mlp_old_input_layer->output_layer = Qnil;
  }

  mlp_layer->narr_input = val_input;
  mlp_layer->input_layer = Qnil;
  p_mlp_layer_run( mlp_layer );

  layer_object = mlp_layer->output_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
    p_mlp_layer_run( mlp_layer );
  }

  return mlp_layer->narr_output;
}

VALUE mlp_network_object_ms_error( VALUE self, VALUE target ) {
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  volatile VALUE val_target;
  VALUE layer_object;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  val_target = na_cast_object(target, NA_SFLOAT);
  GetNArray( val_target, na_target );

  if ( na_target->rank != 1 ) {
    rb_raise( rb_eArgError, "Target rank should be 1, but got %d", na_target->rank );
  }

  layer_object = mlp_network->first_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  if ( na_target->total != mlp_layer->num_outputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match network output size %d", na_target->total, mlp_layer->num_outputs );
  }

  GetNArray( mlp_layer->narr_output, na_output );

  return FLT2NUM( core_mean_square_error( mlp_layer->num_outputs, (float *) na_output->ptr,  (float *) na_target->ptr ) );
}



VALUE mlp_network_object_train_once( VALUE self, VALUE new_input, VALUE target ) {
  struct NARRAY *na_input;
  volatile VALUE val_input;
  struct NARRAY *na_target;
  struct NARRAY *na_output;
  struct NARRAY *na_output_slope;
  struct NARRAY *na_output_deltas;

  volatile VALUE val_target;
  volatile VALUE layer_object;

  MLP_Layer *mlp_old_input_layer;
  MLP_Layer *mlp_layer;
  MLP_Network *mlp_network = get_mlp_network_struct( self );

  ////////////////////////////////////////////////////////////////////////////////////
  // Check input is valid
  layer_object = mlp_network->first_layer;
  Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );

  val_input = na_cast_object(new_input, NA_SFLOAT);
  GetNArray( val_input, na_input );

  if ( na_input->rank != 1 ) {
    rb_raise( rb_eArgError, "Inputs rank should be 1, but got %d", na_input->rank );
  }

  if ( na_input->total != mlp_layer->num_inputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match layer input size %d", na_input->total, mlp_layer->num_inputs );
  }


  ////////////////////////////////////////////////////////////////////////////////////
  // Check target is valid
  val_target = na_cast_object(target, NA_SFLOAT);
  GetNArray( val_target, na_target );

  if ( na_target->rank != 1 ) {
    rb_raise( rb_eArgError, "Target rank should be 1, but got %d", na_target->rank );
  }

  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
  }

  if ( na_target->total != mlp_layer->num_outputs ) {
    rb_raise( rb_eArgError, "Array size %d does not match network output size %d", na_target->total, mlp_layer->num_outputs );
  }

  ////////////////////////////////////////////////////////////////////////////////////
  // Attach input
  layer_object = mlp_network->first_layer;
  Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
  if ( ! NIL_P( mlp_layer->input_layer ) ) {
    // This layer has an existing input layer, it needs to stop pointing its output here
    Data_Get_Struct( mlp_layer->input_layer, MLP_Layer, mlp_old_input_layer );
    mlp_old_input_layer->output_layer = Qnil;
  }
  mlp_layer->narr_input = val_input;
  mlp_layer->input_layer = Qnil;

  ////////////////////////////////////////////////////////////////////////////////////
  // Run the network forward
  p_mlp_layer_run( mlp_layer );
  layer_object = mlp_layer->output_layer;
  while ( ! NIL_P(layer_object) ) {
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
    layer_object = mlp_layer->output_layer;
    p_mlp_layer_run( mlp_layer );
  }

  ////////////////////////////////////////////////////////////////////////////////////
  // Calculate delta on output
  GetNArray( mlp_layer->narr_output, na_output );
  GetNArray( mlp_layer->narr_output_slope, na_output_slope );
  GetNArray( mlp_layer->narr_output_deltas, na_output_deltas );

  transfer_bulk_derivative_at( mlp_layer->transfer_fn, mlp_layer->num_outputs, (float *) na_output->ptr, (float *) na_output_slope->ptr );

  core_calc_output_deltas( mlp_layer->num_outputs, (float *) na_output->ptr,
      (float *) na_output_slope->ptr, (float *) na_target->ptr, (float *) na_output_deltas->ptr );

  ////////////////////////////////////////////////////////////////////////////////////
  // Back-propagate delta

  while ( ! NIL_P(mlp_layer->input_layer) ) {

    layer_object = mlp_layer->input_layer;
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
  }

  ////////////////////////////////////////////////////////////////////////////////////
  // Adjust weights

  while ( ! NIL_P(mlp_layer->output_layer) ) {

    layer_object = mlp_layer->output_layer;
    Data_Get_Struct( layer_object, MLP_Layer, mlp_layer );
  }

  ////////////////////////////////////////////////////////////////////////////////////
  // Return ms_error

  GetNArray( mlp_layer->narr_output, na_output );

  return FLT2NUM( core_mean_square_error( mlp_layer->num_outputs, (float *) na_output->ptr,  (float *) na_target->ptr ) );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mlp_network_class( VALUE parent_module ) {
  // Layer instantiation and class methods
  rb_define_alloc_func( Network, mlp_network_alloc );
  rb_define_method( Network, "initialize", mlp_network_class_initialize, 3 );
  rb_define_method( Network, "initialize_copy", mlp_network_class_initialize_copy, 1 );

  // Network attributes
  rb_define_method( Network, "num_layers", mlp_network_object_num_layers, 0 );
  rb_define_method( Network, "num_inputs", mlp_network_object_num_inputs, 0 );
  rb_define_method( Network, "num_outputs", mlp_network_object_num_outputs, 0 );
  rb_define_method( Network, "input", mlp_network_object_input, 0 );
  rb_define_method( Network, "output", mlp_network_object_output, 0 );
  rb_define_method( Network, "layers", mlp_network_object_layers, 0 );

  // Network methods
  rb_define_method( Network, "init_weights", mlp_network_object_init_weights, -1 );
  rb_define_method( Network, "run", mlp_network_object_run, 1 );
  rb_define_method( Network, "ms_error", mlp_network_object_ms_error, 1 );
  rb_define_method( Network, "train_once", mlp_network_object_train_once, 2 );
}
