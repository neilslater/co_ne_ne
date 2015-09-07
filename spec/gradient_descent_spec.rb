require 'helpers'

class TestQuadratic
  attr_reader :roots, :factors

  def initialize nparams
    @roots = ( NArray.sfloat( nparams ).random  - 0.5 ) * 10
    @factors = ( NArray.sfloat( nparams ).random + 0.1 ) * 3
  end

  def value_at params
    ( (params - @roots) * (params - @roots) * @factors ).sum
  end

  def gradients_at params
    2 * @factors * ( params  - @roots )
  end
end

describe RuNeNe::GradientDescent::SGD do
  describe "class methods" do
    let(:example_params) { NArray.sfloat(2) }

    describe "#new" do
      it "creates a new object" do
        expect( RuNeNe::GradientDescent::SGD.new( example_params ) ).to be_a RuNeNe::GradientDescent::SGD
      end

      it "should take number of params from example input" do
        gd = RuNeNe::GradientDescent::SGD.new( example_params )
        expect( gd.num_params ).to be 2
      end

      it "should take number of params regardless of rank" do
        gd = RuNeNe::GradientDescent::SGD.new( NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' ) )
        expect( gd.num_params ).to be 16
      end
    end

    describe "with Marshal" do
      before do
        @orig_data = RuNeNe::GradientDescent::SGD.new( example_params )
        @saved_data = Marshal.dump( @orig_data )
        @copy_data =  Marshal.load( @saved_data )
      end

      it "can save and retrieve gradient descent settings" do
        expect( @copy_data ).to_not be @orig_data
        expect( @copy_data.num_params ).to be 2
      end
    end
  end

  describe "instance methods" do
    let :gd do
       RuNeNe::GradientDescent::SGD.new( NArray.sfloat(2) )
    end

    before :each do
      NArray.srand(555)
      @params = (NArray.sfloat(2).random - 0.5) * 10
    end

    describe "#clone" do
      it "should make a simple copy of number of params" do
        copy = gd.clone
        expect( copy.num_params ).to eql gd.num_params
      end
    end

    describe "#pre_gradient_step" do
      it "should not alter params" do
        before_params = @params.clone
        gd.pre_gradient_step( @params, 1.0 )
        expect( @params ).to be_narray_like( before_params, 1e-16 )
      end
    end

    describe "#gradient_step" do
      it "should alter params" do
        before_params = @params.clone
        gradients = NArray.sfloat(2).random - 0.5
        gd.gradient_step( @params, gradients, 0.1)
        expect( @params ).to_not be_narray_like before_params
      end
    end

    describe "optimisation" do
      it "should optimise a simple quadratic" do
        tq = TestQuadratic.new(2)

        # Pre-optimisation check that there is something to optimise
        expect( tq.value_at( @params ) ).to_not be_within(0.1).of 0
        expect( @params ).to_not be_narray_like tq.roots

        20.times do
          gd.pre_gradient_step( @params, 0.1 )
          gd.gradient_step( @params, tq.gradients_at(@params), 0.1 )
        end
        expect( tq.value_at( @params ) ).to be_within(1e-6).of 0
        expect( @params ).to be_narray_like tq.roots
      end
    end
  end
end

describe RuNeNe::GradientDescent::NAG do
  describe "class methods" do
    let(:example_params) { NArray.sfloat(2) }

    describe "#new" do
      it "creates a new object" do
        expect( RuNeNe::GradientDescent::NAG.new( example_params, 0.9 ) ).to be_a RuNeNe::GradientDescent::NAG
      end

      it "should take number of params from example input" do
        gd = RuNeNe::GradientDescent::NAG.new( example_params, 0.9 )
        expect( gd.num_params ).to be 2
      end

      it "should take number of params regardless of rank" do
        gd = RuNeNe::GradientDescent::NAG.new( NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' ), 0.9 )
        expect( gd.num_params ).to be 16
      end

      it "should set a momentum value" do
        gd = RuNeNe::GradientDescent::NAG.new( example_params, 0.9 )
        expect( gd.momentum ).to be_within(1e-6).of 0.9
      end

      it "should initialize a weight_velocity array to match params size and shape" do
        gd = RuNeNe::GradientDescent::NAG.new( example_params, 0.9 )
        expect( gd.weight_velocity ).to be_narray_like NArray[0.0, 0.0]

        gd = RuNeNe::GradientDescent::NAG.new( NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' ), 0.9 )
        expect( gd.weight_velocity ).to be_narray_like NArray.cast( [ [ [0.0, 0.0], [0.0, 0.0] ],
            [ [0.0, 0.0], [0.0, 0.0] ], [ [ 0.0, 0.0], [0.0, 0.0] ],
            [ [ 0.0, 0.0], [0.0, 0.0] ] ], 'sfloat' )
      end
    end

    describe "with Marshal" do
      before do
        @orig_data = RuNeNe::GradientDescent::NAG.new( example_params, 0.9 )
        @orig_data.weight_velocity[0..1] = NArray[0.3,-0.3]

        @saved_data = Marshal.dump( @orig_data )
        @copy_data =  Marshal.load( @saved_data )
      end

      it "can save and retrieve gradient descent settings" do
        expect( @copy_data ).to_not be @orig_data
        expect( @copy_data.num_params ).to be 2
        expect( @copy_data.momentum ).to be_within(1e-6).of 0.9
      end

      it "can save and retrieve weight_velocities" do
        expect( @copy_data ).to_not be @orig_data
        expect( @copy_data.weight_velocity ).to_not be @orig_data.weight_velocity
        expect( @copy_data.weight_velocity ).to be_narray_like @orig_data.weight_velocity
      end
    end
  end

  describe "instance methods" do
    let :gd do
       RuNeNe::GradientDescent::NAG.new( NArray.sfloat(2), 0.9 )
    end

    before :each do
      NArray.srand(555)
      @params = (NArray.sfloat(2).random - 0.5) * 10
    end

    describe "#clone" do
      it "should make a simple copy of number of params" do
        copy = gd.clone
        expect( copy.num_params ).to eql gd.num_params
      end
    end

    describe "#pre_gradient_step" do
      it "should not alter params" do
        before_params = @params.clone
        gd.pre_gradient_step( @params, 1.0 )
        expect( @params ).to be_narray_like( before_params, 1e-16 )
      end
    end

    describe "#gradient_step" do
      it "should alter params" do
        before_params = @params.clone
        gradients = NArray.sfloat(2).random - 0.5
        gd.gradient_step( @params, gradients, 0.1)
        expect( @params ).to_not be_narray_like before_params
      end
    end

    describe "optimisation" do
      it "should optimise a simple quadratic" do
        tq = TestQuadratic.new(2)

        # Pre-optimisation check that there is something to optimise
        expect( tq.value_at( @params ) ).to_not be_within(0.1).of 0
        expect( @params ).to_not be_narray_like tq.roots

        20.times do
          gd.pre_gradient_step( @params, 0.1 )
          gd.gradient_step( @params, tq.gradients_at(@params), 0.1 )
        end
        expect( tq.value_at( @params ) ).to be_within(1e-6).of 0
        expect( @params ).to be_narray_like tq.roots
      end
    end
  end
end
