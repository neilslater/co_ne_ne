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
