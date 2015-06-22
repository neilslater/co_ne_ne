// ext/ru_ne_ne/ruby_module_objective.c

#include "ruby_module_objective.h"

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

////////////////////////////////////////////////////////////////////////////////////////////////////

void init_objective_module( ) {
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "loss", mse_loss, 2 );
  rb_define_singleton_method( RuNeNe_Objective_MeanSquaredError, "delta_loss", mse_delta_loss, 2 );
}
