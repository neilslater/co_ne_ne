// ext/ru_ne_ne/ruby_module_objective.c

#include "ruby_module_objective.h"

// Pointer types for generic loss function
typedef float (*loss_fn)(int n, float * p, float * t);
typedef void (*delta_loss_fn)(int n, float * p, float * t, float * d);

// Following wrappers define functions so that they can be used with function pointers

#define ETA 1e-15

float wrapped_logloss( int n, float* predictions, float* targets ) {
  return raw_logloss( n, predictions, targets, ETA );
}

void wrapped_delta_logloss( int n, float* predictions, float* targets, float* delta_loss ) {
  raw_delta_logloss( n, predictions, targets, delta_loss, ETA );
  return;
}

float wrapped_mlogloss( int n, float* predictions, float* targets ) {
  return raw_mlogloss( n, predictions, targets, ETA );
}

void wrapped_delta_mlogloss( int n, float* predictions, float* targets, float* delta_loss ) {
  raw_delta_mlogloss( n, predictions, targets, delta_loss, ETA );
  return;
}

static VALUE generic_loss_function( VALUE rv_predictions, VALUE rv_targets, loss_fn fn ) {
  volatile VALUE val_predictions;
  volatile VALUE val_targets;
  struct NARRAY *na_predictions;
  struct NARRAY *na_targets;

  val_predictions = na_cast_object( rv_predictions, NA_SFLOAT );
  GetNArray( val_predictions, na_predictions );

  val_targets = na_cast_object( rv_targets, NA_SFLOAT );
  GetNArray( val_targets, na_targets );

  if ( na_predictions->total !=  na_targets->total ) {
    rb_raise( rb_eArgError, "Predictions length %d, but targets length %d", na_predictions->total, na_targets->total );
  }

  return FLT2NUM( fn( na_predictions->total, (float*) na_predictions->ptr, (float*) na_targets->ptr ) );
}

static VALUE generic_delta_loss_function( VALUE rv_predictions, VALUE rv_targets, delta_loss_fn fn ) {
  volatile VALUE val_predictions;
  volatile VALUE val_targets;
  struct NARRAY *na_predictions;
  struct NARRAY *na_targets;

  val_predictions = na_cast_object( rv_predictions, NA_SFLOAT );
  GetNArray( val_predictions, na_predictions );

  val_targets = na_cast_object( rv_targets, NA_SFLOAT );
  GetNArray( val_targets, na_targets );

  if ( na_predictions->total !=  na_targets->total ) {
    rb_raise( rb_eArgError, "Predictions length %d, but targets length %d", na_predictions->total, na_targets->total );
  }

  struct NARRAY *na_delta_loss;
  volatile VALUE rv_delta_loss = na_make_object( NA_SFLOAT, na_predictions->rank, na_predictions->shape, cNArray );
  GetNArray( rv_delta_loss, na_delta_loss );

  fn( na_predictions->total, (float*) na_predictions->ptr, (float*) na_targets->ptr, (float*) na_delta_loss->ptr );

  return rv_delta_loss;
}


/* Document-module:  RuNeNe::Objective::MeanSquaredError
 *
 * The mean squared error function is a common choice for regression problems.
 */

/* @overload loss( predictions, targets )
 * Calculates a single example row's contributions to mean squared error loss, equivalent to Ruby code
 *     0.5 * ( predictions.zip( targets ).inject(0) { |sum,pt| sum + (pt.first-pt.last)**2 } )
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [Float] loss for the example
 */
static VALUE mse_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_loss_function( rv_predictions, rv_targets, raw_mse_loss );
}

/* @overload delta_loss( predictions, targets )
 * Calculates the partial derivative of the loss value with respect to each prediction.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt predictions
 */
static VALUE mse_delta_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_delta_loss_function( rv_predictions, rv_targets, raw_mse_delta_loss );
}

/* @overload linear_de_dz( predictions, targets )
 * Calculates the partial derivative of the loss value with respect to z value from before the
 * linear transfer function for given predictions and targets. Identical in practice to delta_loss
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt predictions
 */
static VALUE mse_linear_de_dz( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_delta_loss_function( rv_predictions, rv_targets, obj_mse_tr_linear_de_dz );
}

/* @overload sigmoid_de_dz( predictions, targets )
 * Calculates the partial derivative of the loss value with respect to z value from before the
 * sigmoid transfer function for given predictions and targets.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt pre-transfer values
 */
static VALUE mse_sigmoid_de_dz( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_delta_loss_function( rv_predictions, rv_targets, obj_mse_tr_sigmoid_de_dz );
}

/* @overload tanh_de_dz( predictions, targets )
 * Calculates the partial derivative of the loss value with respect to z value from before the
 * tanh transfer function for given predictions and targets.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt pre-transfer values
 */
static VALUE mse_tanh_de_dz( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_delta_loss_function( rv_predictions, rv_targets, obj_mse_tr_tanh_de_dz );
}

/* @overload relu_de_dz( predictions, targets )
 * Calculates the partial derivative of the loss value with respect to z value from before the
 * relu transfer function for given predictions and targets.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt pre-transfer values
 */
static VALUE mse_relu_de_dz( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_delta_loss_function( rv_predictions, rv_targets, obj_mse_tr_relu_de_dz );
}

/* @overload softmax_de_dz( predictions, targets )
 * Calculates the partial derivative of the loss value with respect to z value from before the
 * relu transfer function for given predictions and targets.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt pre-transfer values
 */
static VALUE mse_softmax_de_dz( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_delta_loss_function( rv_predictions, rv_targets, obj_mse_tr_softmax_de_dz );
}

/* Document-module:  RuNeNe::Objective::LogLoss
 *
 * Common choice for binary classification outputs, this objective function should only be used
 * when target values are 0 or 1, and outputs are constrained to [0.0,1.0]. Can be used with
 * multiple non-exclusive classes.
 */

/* @overload loss( predictions, targets )
 * Calculates a single example row's contributions to log loss, equivalent to Ruby code
 *     -1.0 * predictions.zip( targets ).inject(0) { |p,t| t * Math.log(p) + (1-t) * Math.log(1-p) }
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [Float] loss for the example
 */
static VALUE logloss_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_loss_function( rv_predictions, rv_targets, wrapped_logloss );
}

/* @overload delta_loss( x )
 * Calculates the partial derivative of the loss value with respect to each prediction.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt predictions
 */
static VALUE logloss_delta_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_delta_loss_function( rv_predictions, rv_targets, wrapped_delta_logloss );
}

/* Document-module:  RuNeNe::Objective::MulticlassLogLoss
 *
 * Common choice for multiple exclusive classification outputs, this objective function should only be used
 * when target values are 0 or 1, and only one target in the set can be 1. It is standard pracice to pair this
 * objective with an output layer that represents mutually-exclusive probabilities, such as softmax.
 */

/* @overload loss( predictions, targets )
 * Calculates a single example row's contributions to log loss, equivalent to Ruby code
 *     -1.0 * predictions.zip( targets ).inject(0) { |p,t| t * Math.log(p) }
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [Float] loss for the example
 */
static VALUE mlogloss_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
    return generic_loss_function( rv_predictions, rv_targets, wrapped_mlogloss );
}

/* @overload delta_loss( x )
 * Calculates the partial derivative of the loss value with respect to each prediction.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt predictions
 */
static VALUE mlogloss_delta_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
  return generic_delta_loss_function( rv_predictions, rv_targets, wrapped_delta_mlogloss );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_objective_module( ) {
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "loss", mse_loss, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "delta_loss", mse_delta_loss, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "linear_de_dz", mse_linear_de_dz, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "sigmoid_de_dz", mse_sigmoid_de_dz, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "tanh_de_dz", mse_tanh_de_dz, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "relu_de_dz", mse_relu_de_dz, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "softmax_de_dz", mse_softmax_de_dz, 2 );


  rb_define_singleton_method( RuNeNe_Objective_LogLoss, "loss", logloss_loss, 2 );
  rb_define_singleton_method( RuNeNe_Objective_LogLoss, "delta_loss", logloss_delta_loss, 2 );

  rb_define_singleton_method( RuNeNe_Objective_MulticlassLogLoss, "loss", mlogloss_loss, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MulticlassLogLoss, "delta_loss", mlogloss_delta_loss, 2 );
}
