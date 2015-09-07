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

      it "should initialize a param_update_velocity array to match params size and shape" do
        gd = RuNeNe::GradientDescent::NAG.new( example_params, 0.9 )
        expect( gd.param_update_velocity ).to be_narray_like NArray[0.0, 0.0]

        gd = RuNeNe::GradientDescent::NAG.new( NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' ), 0.9 )
        expect( gd.param_update_velocity ).to be_narray_like NArray.cast( [ [ [0.0, 0.0], [0.0, 0.0] ],
            [ [0.0, 0.0], [0.0, 0.0] ], [ [ 0.0, 0.0], [0.0, 0.0] ],
            [ [ 0.0, 0.0], [0.0, 0.0] ] ], 'sfloat' )
      end
    end

    describe "with Marshal" do
      before do
        @orig_data = RuNeNe::GradientDescent::NAG.new( example_params, 0.9 )
        @orig_data.param_update_velocity[0..1] = NArray[0.3,-0.3]

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
        expect( @copy_data.param_update_velocity ).to_not be @orig_data.param_update_velocity
        expect( @copy_data.param_update_velocity ).to be_narray_like @orig_data.param_update_velocity
      end
    end
  end

  describe "instance methods" do
    let :gd do
       # Low momentum here, otherwise on this very smooth test surface, the optimiser oscillates
       RuNeNe::GradientDescent::NAG.new( NArray.sfloat(2), 0.5 )
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
      it "should not alter params on first ever step" do
        before_params = @params.clone
        gd.pre_gradient_step( @params, 1.0 )
        expect( @params ).to be_narray_like( before_params, 1e-16 )
      end

      it "should alter params on second step due to momentum" do
        before_params = @params.clone
        gd.pre_gradient_step( @params, 1.0 )
        gradients = NArray.sfloat(2).random - 0.5
        gd.gradient_step( @params, gradients, 0.1)

        after_first_step_params = @params.clone

        gd.pre_gradient_step( @params, 1.0 )
        expect( @params ).to_not be_narray_like after_first_step_params
      end
    end

    describe "#gradient_step" do
      it "should alter params" do
        before_params = @params.clone
        gradients = NArray.sfloat(2).random - 0.5
        gd.gradient_step( @params, gradients, 0.1)
        expect( @params ).to_not be_narray_like before_params
      end

      it "should alter param_update_velocity" do
        gradients = NArray.sfloat(2).random - 0.5
        gd.gradient_step( @params, gradients, 0.1)
        expect( gd.param_update_velocity ).to_not be_narray_like NArray[0.0,0.0]
      end
    end

    describe "optimisation" do
      it "should optimise a simple quadratic" do
        tq = TestQuadratic.new(2)

        # Pre-optimisation check that there is something to optimise
        expect( tq.value_at( @params ) ).to_not be_within(0.1).of 0
        expect( @params ).to_not be_narray_like tq.roots

        # On the simple quadratic surface, Nesterov momentum is almost a liability
        # Hence additonal 10 iterations required compared to plan SGD
        30.times do
          gd.pre_gradient_step( @params, 0.03 )
          gd.gradient_step( @params, tq.gradients_at(@params), 0.03 )
        end
        expect( tq.value_at( @params ) ).to be_within(1e-6).of 0
        expect( @params ).to be_narray_like tq.roots
      end
    end
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

describe RuNeNe::GradientDescent::RMSProp do
  describe "class methods" do
    let(:example_params) { NArray.sfloat(2) }

    describe "#new" do
      it "creates a new object" do
        expect( RuNeNe::GradientDescent::RMSProp.new( example_params, 0.9, 1e-6 ) ).to be_a RuNeNe::GradientDescent::RMSProp
      end

      it "should take number of params from example input" do
        gd = RuNeNe::GradientDescent::RMSProp.new( example_params, 0.9, 1e-6 )
        expect( gd.num_params ).to be 2
      end

      it "should take number of params regardless of rank" do
        gd = RuNeNe::GradientDescent::RMSProp.new( NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' ), 0.9, 1e-6 )
        expect( gd.num_params ).to be 16
      end

      it "should set a decay value" do
        gd = RuNeNe::GradientDescent::RMSProp.new( example_params, 0.9, 1e-6 )
        expect( gd.decay ).to be_within(1e-6).of 0.9
      end

      it "should set an epsilon value" do
        gd = RuNeNe::GradientDescent::RMSProp.new( example_params, 0.9, 1e-6 )
        expect( gd.epsilon ).to be_within(1e-12).of 1e-6
      end

      it "should initialize a av_squared_grads array to match params size and shape" do
        gd = RuNeNe::GradientDescent::RMSProp.new( example_params, 0.9, 1e-6 )
        expect( gd.av_squared_grads  ).to be_narray_like NArray[1.0, 1.0]

        gd = RuNeNe::GradientDescent::RMSProp.new( NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' ), 0.9, 1e-6 )
        expect( gd.av_squared_grads ).to be_narray_like NArray.cast( [ [ [1.0, 1.0], [1.0, 1.0] ],
            [ [1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, 1.0], [1.0, 1.0] ],
            [ [ 1.0, 1.0], [1.0, 1.0] ] ], 'sfloat' )
      end
    end

    describe "with Marshal" do
      before do
        @orig_data = RuNeNe::GradientDescent::RMSProp.new( example_params, 0.9, 1e-6 )
        @orig_data.av_squared_grads[0..1] = NArray[0.3,1.5]

        @saved_data = Marshal.dump( @orig_data )
        @copy_data =  Marshal.load( @saved_data )
      end

      it "can save and retrieve gradient descent settings" do
        expect( @copy_data ).to_not be @orig_data
        expect( @copy_data.num_params ).to be 2
        expect( @copy_data.decay ).to be_within(1e-6).of 0.9
        expect( @copy_data.epsilon ).to be_within(1e-12).of 1e-6
      end

      it "can save and retrieve av_squared_grads" do
        expect( @copy_data ).to_not be @orig_data
        expect( @copy_data.av_squared_grads ).to_not be @orig_data.av_squared_grads
        expect( @copy_data.av_squared_grads ).to be_narray_like @orig_data.av_squared_grads
      end
    end
  end

  describe "instance methods" do
    let :gd do
       RuNeNe::GradientDescent::RMSProp.new( NArray.sfloat(2), 0.9, 1e-6 )
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
      it "should not alter params on first ever step" do
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

      it "should alter av_squared_grads" do
        gradients = NArray.sfloat(2).random - 0.5
        before_av_squared_grads = gd.av_squared_grads.clone
        gd.gradient_step( @params, gradients, 0.1)
        expect( gd.av_squared_grads ).to_not be_narray_like before_av_squared_grads
      end
    end

    describe "optimisation" do
      it "should optimise a simple quadratic" do
        tq = TestQuadratic.new(2)

        # Pre-optimisation check that there is something to optimise
        expect( tq.value_at( @params ) ).to_not be_within(0.1).of 0
        expect( @params ).to_not be_narray_like tq.roots

        # RMSProp seems to cope very badly with the high gradients here
        70.times do
          gd.pre_gradient_step( @params, 0.2 )
          gd.gradient_step( @params, tq.gradients_at(@params), 0.2 )
        end
        expect( tq.value_at( @params ) ).to be_within(1e-6).of 0
        expect( @params ).to be_narray_like tq.roots
      end
    end
  end
end
