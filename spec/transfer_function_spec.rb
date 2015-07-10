require 'helpers'

[ RuNeNe::Transfer::Sigmoid, RuNeNe::Transfer::TanH,
  RuNeNe::Transfer::ReLU, RuNeNe::Transfer::Linear ].each do |transfer|
describe transfer do
  let( :x_vals ) { (-100..100).to_a.map { |x| x.to_f/10 } }
  let( :big_array ) { NArray.sfloat(20,20).random(2.0) - 1.0 }
  let( :original_array ) { big_array.clone }

  describe "#function" do
    it "outputs a smallish float value for a smallish input value" do
      x_vals.each do |x|
        y = transfer.function(x)
        expect( y ).to be_a Float
        expect( y ).to be_within(20.0).of x
      end
    end

    it "ouputs same or larger y for increasing x" do
      max_so_far = -100.0
      x_vals.each do |x|
        y = transfer.function(x)
        expect( y ).to be >= max_so_far
        max_so_far = y
      end
    end
  end

  describe "#bulk_apply_function" do
    it "alters the whole input narray" do
      expect( big_array ).to be_narray_like original_array
      transfer.bulk_apply_function( big_array )

      unless transfer == RuNeNe::Transfer::Linear
        expect( big_array ).to_not be_narray_like original_array
      end

      20.times do |i|
        20.times do |j|
          expect( big_array[i,j] ).to be_within(1e-6).of transfer.function( original_array[i,j] )
        end
      end
    end
  end

  def approx_dy_dx t, x
    # This is numerical approximation of derivative
    d = 5e-6
    x1 = x + d
    x2 = x - d

    case t.name
    when 'RuNeNe::Transfer::Sigmoid'

      100000.0 * ( 1.0 / ( 1.0 + Math.exp( -x1 ) ) - 1.0 / ( 1.0 + Math.exp( -x2 ) ) )
    when 'RuNeNe::Transfer::TanH'
      100000.0 * ( 2.0 / (1.0 + Math.exp(-2*x1) ) -  2.0 / (1.0 + Math.exp(-2*x2) ) );
    when 'RuNeNe::Transfer::ReLU'
      # For ReLU, dy_dx not well-defined at origin, but we have chosen 0.0
      if x == 0
        0.0
      else
        100000.0 * ( ( x1 > 0.0 ? x1 : 0.0 ) - ( x2 > 0.0 ? x2 : 0.0 ) );
      end
    when 'RuNeNe::Transfer::Linear'
      1.0
    end
  end

  describe "#derivative" do
    it "should return function slope for given x value" do
      x_vals.each do |x|
        dy_dx = transfer.derivative(x)
        expect( dy_dx ).to be_a Float
        expect( dy_dx ).to be_within(1e-4).of approx_dy_dx( transfer, x )
      end
    end
  end

  describe "#derivative_at" do
    it "should return function slope for given y value" do
      x_vals.each do |x|
        y = transfer.function(x)
        dy_dx = transfer.derivative_at(y)
        expect( dy_dx ).to be_a Float
        expect( dy_dx ).to be_within(1e-4).of approx_dy_dx( transfer, x )
      end
    end
  end
end
end

describe "Output value tests for" do
  let( :test_array ) { NArray.cast( [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
      0.1, 0.2, 0.3, 0.4, 0.5], 'sfloat' ) }

  describe RuNeNe::Transfer::Sigmoid do
    it "should match normal definition of sigmoid function" do
      RuNeNe::Transfer::Sigmoid.bulk_apply_function( test_array )
      expect( test_array ).to be_narray_like NArray.cast( [ 0.377541, 0.401312, 0.425557, 0.450166,
          0.475021, 0.5, 0.524979, 0.549834, 0.574443, 0.598688, 0.622459], 'sfloat' )
    end
  end

  describe RuNeNe::Transfer::TanH do
    it "should match normal definition of tanh function" do
      RuNeNe::Transfer::TanH.bulk_apply_function( test_array )
      expect( test_array ).to be_narray_like NArray.cast( [ -0.462117, -0.379949, -0.291313, -0.197375,
          -0.099668, 0.0, 0.099668, 0.197375, 0.291313, 0.379949, 0.462117 ], 'sfloat' )
    end
  end

  describe RuNeNe::Transfer::ReLU do
    it "should match normal definition of 'rectified linear' function" do
      RuNeNe::Transfer::ReLU.bulk_apply_function( test_array )
      expect( test_array ).to be_narray_like NArray.cast( [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.1, 0.2, 0.3, 0.4, 0.5], 'sfloat' )
    end
  end

  describe RuNeNe::Transfer::Linear do
    it "should not change values" do
      RuNeNe::Transfer::Linear.bulk_apply_function( test_array )
      expect( test_array ).to be_narray_like NArray.cast( [ -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
          0.1, 0.2, 0.3, 0.4, 0.5], 'sfloat' )
    end
  end
end


describe RuNeNe::Transfer::Softmax do
  describe "#bulk_apply_function" do
    it "alters the whole input narray" do
      10.times do |n|
        test_array = NArray.sfloat( n + 2 ).random(2.0) - 1.0
        original_array = test_array.clone
        expect( test_array ).to be_narray_like original_array
        RuNeNe::Transfer::Softmax.bulk_apply_function( test_array )
        expect( test_array ).to_not be_narray_like original_array

        # Specific to Softmax, probabilities should sum to 1.0
        expect( test_array.sum ).to be_within(1e-6).of 1.0
      end
    end
  end

  def approx_dy_dx orig_inputs
    # This is numerical approximation of derivative
    d = 5e-4
    s = orig_inputs.size
    dy_dx = NArray.sfloat( s, s )

    (0...s).each do |k|
      # We figure out dy(i)_dx(k)
      up_values = orig_inputs.clone
      up_values[k] += d
      RuNeNe::Transfer::Softmax.bulk_apply_function( up_values )

      down_values = orig_inputs.clone
      down_values[k] -= d
      RuNeNe::Transfer::Softmax.bulk_apply_function( down_values )
      dy_dx[k,(0...s)] = (up_values - down_values) / ( 2 * d )
    end

    dy_dx
  end

  describe "#bulk_derivative_at" do
    it "should return function slope matrix" do
      5.times do |n|
        orig_inputs = NArray.sfloat( n + 2 ).random(2.0) - 1.0

        # It's a matrix because each input value affects every output value in a different way
        rough_gradient = approx_dy_dx( orig_inputs )

        fn_values = orig_inputs.clone
        RuNeNe::Transfer::Softmax.bulk_apply_function( fn_values )
        expect( RuNeNe::Transfer::Softmax.bulk_derivative_at( fn_values ) ).to be_narray_like( rough_gradient, 1e-6 )
      end
    end
  end
end
