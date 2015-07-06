// ext/ru_ne_ne/ruby_module_objective.c

#include "ruby_module_objective.h"

/* TODO: Use a C function pointer in place of void* fun
static VALUE generic_loss_function( VALUE rv_predictions, VALUE rv_targets, void* fun ) {
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

  return FLT2NUM( fun( na_predictions->total, (float*) na_predictions->ptr, (float*) na_targets->ptr ) );
}
*/

/* Document-module:  RuNeNe::Objective::MeanSquaredError
 *
 * The mean squared error function is a common choice for regression problems.
 */

/* @overload loss( predictions, targets )
 * Calculates a single example row's contributions to mean squared error loss, equivalent to Ruby code
 *     0.5 * ( predictions.zip( targets ).inject(0) { |p,t| (p-t)**2 } )
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [Float] loss for the example
 */
static VALUE mse_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
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

  return FLT2NUM( raw_mse_loss( na_predictions->total, (float*) na_predictions->ptr, (float*) na_targets->ptr ) );
}

/* @overload delta_loss( x )
 * Calculates the partial derivative of the loss value with respect to each prediction.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt predictions
 */
static VALUE mse_delta_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
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

  raw_mse_delta_loss( na_predictions->total, (float*) na_predictions->ptr, (float*) na_targets->ptr, (float*) na_delta_loss->ptr );

  return rv_delta_loss;
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

  return FLT2NUM( raw_logloss( na_predictions->total, (float*) na_predictions->ptr, (float*) na_targets->ptr, 1e-15 ) );
}

/* @overload delta_loss( x )
 * Calculates the partial derivative of the loss value with respect to each prediction.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt predictions
 */
static VALUE logloss_delta_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
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

  raw_delta_logloss( na_predictions->total, (float*) na_predictions->ptr, (float*) na_targets->ptr, (float*) na_delta_loss->ptr, 1e-15 );

  return rv_delta_loss;
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

  return FLT2NUM( raw_mlogloss( na_predictions->total, (float*) na_predictions->ptr, (float*) na_targets->ptr, 1e-15 ) );
}

/* @overload delta_loss( x )
 * Calculates the partial derivative of the loss value with respect to each prediction.
 * @param [NArray<sfloat>] predictions
 * @param [NArray<sfloat>] targets
 * @return [NArray<sfloat>] partial derivatives of loss wrt predictions
 */
static VALUE mlogloss_delta_loss( VALUE self, VALUE rv_predictions, VALUE rv_targets ) {
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

  raw_delta_mlogloss( na_predictions->total, (float*) na_predictions->ptr, (float*) na_targets->ptr, (float*) na_delta_loss->ptr, 1e-15 );

  return rv_delta_loss;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void init_objective_module( ) {
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "loss", mse_loss, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "delta_loss", mse_delta_loss, 2 );

  rb_define_singleton_method( RuNeNe_Objective_LogLoss, "loss", logloss_loss, 2 );
  rb_define_singleton_method( RuNeNe_Objective_LogLoss, "delta_loss", logloss_delta_loss, 2 );

  rb_define_singleton_method( RuNeNe_Objective_MulticlassLogLoss, "loss", mlogloss_loss, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MulticlassLogLoss, "delta_loss", mlogloss_delta_loss, 2 );
}
