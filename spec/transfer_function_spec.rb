require 'helpers'

def qsig x
  1.0 / ( 1.0 + Math.exp( -x ) )
end

[ CoNeNe::Transfer::Sigmoid, CoNeNe::Transfer::TanH, CoNeNe::Transfer::ReLU ].each do |transfer|
describe transfer do
  let( :x_vals ) { (-100..100).to_a.map { |x| x.to_f/10 } }
  let( :big_array ) { NArray.sfloat(20,20).random(2.0) - 1.0 }
  let( :original_array ) { big_array.clone }

  describe "#function" do
    it "outputs a smallish float value for a smallish input value" do
      x_vals.each do |x|
        y = transfer.function(x)
        y.should be_a Float
        y.should be_within(20.0).of x
      end
    end

    it "ouputs same or larger y for increasing x" do
      max_so_far = -100.0
      x_vals.each do |x|
        y = transfer.function(x)
        y.should be >= max_so_far
        max_so_far = y
      end
    end
  end

  describe "#bulk_apply_function" do
    it "alters the whole input narray" do
      big_array.should be_narray_like original_array
      transfer.bulk_apply_function( big_array )
      big_array.should_not be_narray_like original_array
      20.times do |i|
        20.times do |j|
          big_array[i,j].should be_within(1e-6).of transfer.function( original_array[i,j] )
        end
      end
    end
  end

  def approx_dy_dx t, x
    # For ReLU, dy_dx not well-defined at origin, but we have chosen 0.0
    return 0.0 if x == 0.0 && t == CoNeNe::Transfer::ReLU

    # This is numerical approximation of derivative
    d = 5e-6
    case t.name
    when 'CoNeNe::Transfer::Sigmoid'
      100000.0 * ( qsig( x + d ) - qsig( x - d ) )
    when 'CoNeNe::Transfer::TanH'
      100000.0 * ( t.function( x + d ) - t.function( x - d ) )
    when 'CoNeNe::Transfer::ReLU'
      100000.0 * ( t.function( x + d ) - t.function( x - d ) )
    end
  end

  describe "#derivative" do
    it "should return function slope for given x value" do
      x_vals.each do |x|
        dy_dx = transfer.derivative(x)
        dy_dx.should be_a Float
        dy_dx.should be_within(1e-4).of approx_dy_dx( transfer, x )
      end
    end
  end

  describe "#derivative_at" do
    it "should return function slope for given y value" do
      x_vals.each do |x|
        y = transfer.function(x)
        dy_dx = transfer.derivative_at(y)
        dy_dx.should be_a Float
        dy_dx.should be_within(1e-4).of approx_dy_dx( transfer, x )
      end
    end
  end
end
end

describe "Output value tests for" do
  let( :test_array ) { NArray.cast( [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
      0.1, 0.2, 0.3, 0.4, 0.5], 'sfloat' ) }

  describe CoNeNe::Transfer::Sigmoid do
    it "should match normal definition of sigmoid function" do
      CoNeNe::Transfer::Sigmoid.bulk_apply_function( test_array )
      test_array.should be_narray_like NArray.cast( [ 0.377541, 0.401312, 0.425557, 0.450166,
          0.475021, 0.5, 0.524979, 0.549834, 0.574443, 0.598688, 0.622459], 'sfloat' )
    end
  end

  describe CoNeNe::Transfer::TanH do
    it "should match normal definition of tanh function" do
      CoNeNe::Transfer::TanH.bulk_apply_function( test_array )
      test_array.should be_narray_like NArray.cast( [ -0.462117, -0.379949, -0.291313, -0.197375,
          -0.099668, 0.0, 0.099668, 0.197375, 0.291313, 0.379949, 0.462117 ], 'sfloat' )
    end
  end

  describe CoNeNe::Transfer::ReLU do
    it "should match normal definition of 'rectified linear' function" do
      CoNeNe::Transfer::ReLU.bulk_apply_function( test_array )
      test_array.should be_narray_like NArray.cast( [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.1, 0.2, 0.3, 0.4, 0.5], 'sfloat' )
    end
  end

end
