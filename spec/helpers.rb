# ru_ne_ne/spec/helpers.rb
require 'coveralls'
require 'mocha/api'

Coveralls.wear!

require 'ru_ne_ne'

# Matcher compares NArrays numerically
RSpec::Matchers.define :be_narray_like do |expected_narray,diff_limit=1e-9|
  match do |given|
    @error = nil
    if ! given.is_a?(NArray)
      @error = "Wrong class."
    elsif given.shape != expected_narray.shape
      @error = "Shapes are different."
    else
      d = given - expected_narray
      difference =  ( d * d ).sum / d.size
      if difference > diff_limit
        @error = "Numerical difference with mean square error #{difference}"
      end
    end
    @given = given.clone

    if @error
      @expected = expected_narray.clone
    end

    ! @error
  end

  failure_message do
    "NArray does not match supplied example. #{@error}
    Expected: #{@expected.inspect}
    Got: #{@given.inspect}"
  end

  failure_message_when_negated do
    "NArray is too close to unwanted example.
    Unwanted: #{@given.inspect}"
  end

  description do |given, expected|
    "numerically very close to example"
  end
end

def test_size_range nmin = 1, nmax = 5
  (nmin..nmax).each do |n|
    5.times do
      yield n
    end
  end
end

class TestDemiOutputLayer
  attr_reader :objective, :transfer, :output, :loss

  def initialize objective, transfer
    @objective = objective
    @transfer = transfer
  end

  def run zvals, targets
    @output = transfer.bulk_apply_function( zvals.clone )
    @loss = objective.loss( @output, targets )
  end

  def measure_de_dz zvals, targets, eta = 0.005
    grad_mult = 1.0/(2* eta)
    gradients = zvals.clone

    (0...zvals.size).each do |i|
      up_zvals = zvals.clone
      up_zvals[i] += eta
      down_zvals = zvals.clone
      down_zvals[i] -= eta
      transfer.bulk_apply_function( up_zvals )
      transfer.bulk_apply_function( down_zvals )
      gradients[i] = grad_mult * ( objective.loss( up_zvals, targets ) - objective.loss( down_zvals, targets ) )
    end

    gradients
  end
end

def measure_output_layer_de_dw layer, objective_calc, inputs, targets, eta = 0.005
  grad_mult = 1.0/(2* eta)
  gradients = layer.weights.clone

  (0...gradients.size).each do |i|
    up_layer = layer.clone
    up_layer.weights[i] += eta
    up_outputs = up_layer.run( inputs )
    up_loss = objective_calc.call( up_outputs, targets )

    down_layer = layer.clone
    down_layer.weights[i] -= eta
    down_outputs = down_layer.run( inputs )
    down_loss = objective_calc.call( down_outputs, targets )

    gradients[i] = grad_mult * ( up_loss - down_loss )
  end

  gradients
end

# This de_da is for error grad wrt activations of *inputs* to final layer in the network
def measure_output_layer_de_da layer, objective_calc, inputs, targets, eta = 0.005
  grad_mult = 1.0/(2* eta)
  gradients = inputs.clone

  (0...gradients.size).each do |i|
    up_inputs = inputs.clone
    up_inputs[i] += eta
    up_outputs = layer.run( up_inputs )
    up_loss = objective_calc.call( up_outputs, targets )

    down_inputs = inputs.clone
    down_inputs[i] -= eta
    down_outputs = layer.run( down_inputs )
    down_loss = objective_calc.call( down_outputs, targets )

    gradients[i] = grad_mult * ( up_loss - down_loss )
  end

  gradients
end

# This is essentially a full, but simple, feed-forward network, using Ruby for the
# network architecture and the C-backed classes for the individual layers
class TestLayerStack
  attr_reader :objective, :transfer, :output, :loss, :layers
  attr_reader :training_layers, :input_size, :activations

  # Each layer is given by [output_size, transfer_type]
  def initialize input_size, layers, objective
    @objective = objective
    @input_size = input_size
    @layers = layers.map do |output_size, transfer_type|
      new_layer = RuNeNe::Layer::FeedForward.new( input_size, output_size, transfer_type )
      input_size = output_size
      new_layer
    end
    @training_layers = @layers.map do |rlayer|
      RuNeNe::Trainer::BPLayer.from_layer( rlayer )
    end

    @transfer = transfer
  end

  def start_batch
    @training_layers.each do |bplayer|
      bplayer.start_batch
    end
  end

  def finish_batch
    @training_layers.zip(@layers).each do |bplayer, layer|
      bplayer.finish_batch( layer )
    end
  end

  def process_example inputs, targets
    @activations = [inputs]
    @layers.each do |layer|
      @activations << layer.run( @activations.last )
    end
    @output = @activations.last

    @loss = objective.loss( @output, targets )

    @training_layers.last.backprop_for_output_layer( @layers.last,
        @activations[-2], @activations.last, targets, objective.label )

    layer_ids = [*0..(layers.count-2)]
    layer_ids.reverse.each do |layer_id|
      @training_layers[layer_id].backprop_for_mid_layer( @layers[layer_id],
        @activations[layer_id], @activations[layer_id+1], @training_layers[layer_id+1].de_da )
    end
  end

  def measure_de_dw inputs, targets, eta = 0.005
    de_dw = blank_de_dw
    @layers.each_with_index do |l,i|
      gradients = de_dw[i]
      (0...gradients.size).each do |weight_id|
        gradients[weight_id] = single_weight_de_dw( i, weight_id, eta, inputs, targets )
      end
    end
    de_dw
  end

  private

  def cloned_layers
    @layers.map { |l| l.clone }
  end

  def blank_de_dw
    @layers.map { |l| l.weights * 0 }
  end

  def single_weight_de_dw layer_id, weight_id, eta, inputs, targets
    up_loss = adjusted_w_loss( layer_id, weight_id, eta, inputs, targets )
    down_loss = adjusted_w_loss( layer_id, weight_id, -eta, inputs, targets )
    ( up_loss - down_loss ) / ( 2 * eta )
  end

  def adjusted_w_loss layer_id, weight_id, eta, inputs, targets
    tmp_activations = [inputs]
    tmp_layers = cloned_layers
    tmp_layers[layer_id].weights[weight_id] += eta
    tmp_layers.each do |layer|
      tmp_activations << layer.run( tmp_activations.last )
    end
    objective.loss( tmp_activations.last, targets )
  end
end