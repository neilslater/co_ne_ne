require 'helpers'

describe CoNeNe::MLP::NLayer do
  describe "class methods" do
    describe "#new" do
      it "creates a new layer" do
        CoNeNe::MLP::NLayer.new( 2, 1 ).should be_a CoNeNe::MLP::NLayer
      end

      it "refuses to create new layers for bad parameters" do
        expect { CoNeNe::MLP::NLayer.new( 0, 2 ) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( 3, -1 ) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( "hello", 2 ) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( 3, 2, "garbage" ) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( 3, 2, :foobar ) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( 3, 2, :tanh, 17 ) }.to raise_error
      end

      it "sets values of attributes based on input and output size" do
        CoNeNe.srand( 7000 )

        layer = CoNeNe::MLP::NLayer.new( 3, 2 )
        layer.num_inputs.should == 3
        layer.num_outputs.should == 2
        layer.transfer.should be CoNeNe::Transfer::Sigmoid

        layer.weights.should be_narray_like NArray[
            [ 0.088395, 0.759151, -0.00383174, 0.756848 ],
            [ -0.422237, -0.552743, -0.128862, 0.774815 ] ]

        layer.weights_last_deltas.should be_a NArray
        layer.weights_last_deltas.shape.should == [4,2]

        layer.output.should be_a NArray
        layer.output.shape.should == [2]

        layer.output_deltas.should be_a NArray
        layer.output_deltas.shape.should == [2]
      end

      it "accepts an optional transfer function type param" do
        layer = CoNeNe::MLP::NLayer.new( 4, 1, :sigmoid )
        layer.transfer.should be CoNeNe::Transfer::Sigmoid

        layer = CoNeNe::MLP::NLayer.new( 5, 3, :tanh )
        layer.transfer.should be CoNeNe::Transfer::TanH

        layer = CoNeNe::MLP::NLayer.new( 7, 2, :relu )
        layer.transfer.should be CoNeNe::Transfer::ReLU
      end

      it "plays nicely with Ruby's garbage collection" do
        number_of_layers = 50000

        CoNeNe.srand(800)
        layer = CoNeNe::MLP::NLayer.new( 10, 5 )
        new_layer = nil
        number_of_layers.times do
          new_layer = CoNeNe::MLP::NLayer.new( rand(100)+1, rand(50)+1 )
        end
        GC.start
        sleep 0.5
        layer.output.should be_a NArray
        layer.weights.should be_a NArray
        layer.weights[2,1].should be_within(0.000001).of -0.181608
        number_of_layers.times do
          layer = CoNeNe::MLP::NLayer.new( rand(100)+1, rand(50)+1 )
        end
        sleep 0.5
        new_layer.output.should be_a NArray
        new_layer.weights.should be_a NArray
      end
    end

    describe "#from_weights" do
      it "creates a new layer" do
        CoNeNe::MLP::NLayer.from_weights( NArray.sfloat(4,5) ).should be_a CoNeNe::MLP::NLayer
      end

      it "initialises sizes and output arrays" do
        layer = CoNeNe::MLP::NLayer.from_weights( NArray.sfloat(4,5) )

        layer.num_inputs.should be 3
        layer.num_outputs.should be 5

        layer.output.should be_a NArray
        layer.output.shape.should == [ 5 ]
      end

      it "assigns to the weights attribute directly (not a copy)" do
        w =  NArray.sfloat(12,7)
        layer = CoNeNe::MLP::NLayer.from_weights( w )
        layer.weights.should be w
      end

      it "accepts an optional transfer function type param" do
        w =  NArray.sfloat(3,2)
        layer = CoNeNe::MLP::NLayer.from_weights( w, :sigmoid )
        layer.transfer.should be CoNeNe::Transfer::Sigmoid

        w =  NArray.sfloat(3,2)
        layer = CoNeNe::MLP::NLayer.from_weights( w, :tanh )
        layer.transfer.should be CoNeNe::Transfer::TanH

        w =  NArray.sfloat(3,2)
        layer = CoNeNe::MLP::NLayer.from_weights( w, :relu )
        layer.transfer.should be CoNeNe::Transfer::ReLU
      end

      it "refuses to create new layers for bad parameters" do
        expect { CoNeNe::MLP::NLayer.new( NArray.sfloat(3,2,1) ) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( NArray.sfloat(1,2) ) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( NArray.sfloat(7)) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( NArray.sfloat(5,2), "NOTVALID" ) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( NArray.sfloat(4,1), :blah ) }.to raise_error
        expect { CoNeNe::MLP::NLayer.new( NArray.sfloat(4,1), :tanh, "extras" ) }.to raise_error
      end

    end

  end

  describe "instance methods" do
    let :layer do
      weights = NArray.cast( [ [ -0.1, 0.5, 0.9, 0.7 ], [ -0.6, 0.6, 0.4, 0.6 ] ], 'sfloat' )
      CoNeNe::MLP::NLayer.new( 3, 2 )
      CoNeNe::MLP::NLayer.from_weights( weights )
    end

    describe "#init_weights" do
      before :each do
        CoNeNe.srand(800)
      end

      it "should set weights in range -0.8 to 0.8 by default" do
        layer.init_weights
        layer.weights.should be_narray_like NArray[
          [ 0.458294, -0.067838, -0.342399, 0.455698 ],
          [ 0.790833, -0.181608, 0.752776, 0.1745 ] ]
      end

      it "should accept a single param to set +- range" do
        layer.init_weights( 4.0 )
        layer.weights.should be_narray_like NArray[
          [ 2.29147, -0.33919, -1.712, 2.27849 ],
          [ 3.95417, -0.908039, 3.76388, 0.872502 ] ]
      end

      it "should accept two params to select from a range" do
        layer.init_weights( 0.2, 1.8 )
        layer.weights.should be_narray_like NArray[
          [ 1.45829, 0.932162, 0.657601, 1.4557 ],
          [ 1.79083, 0.818392, 1.75278, 1.1745 ] ]
      end

      it "should work with a negative single param" do
        layer.init_weights( -0.8 )
        layer.weights.should be_narray_like NArray[
          [ -0.458294, 0.067838, 0.342399, -0.455698 ],
          [ -0.790833, 0.181608, -0.752776, -0.1745 ] ]
      end

      it "should work with a 'reversed' range" do
        layer.init_weights( 1.0, 0.0 )
        layer.weights.should be_narray_like NArray[
         [ 0.213566, 0.542399, 0.714, 0.215189 ],
         [ 0.00572914, 0.613505, 0.029515, 0.390937 ] ]
      end

      it "should raise an error for non-numeric params" do
        expect { layer.init_weights( [] ) }.to raise_error
        expect { layer.init_weights( "Hi" ) }.to raise_error
        expect { layer.init_weights( 2.5, :foo => 'bar' ) }.to raise_error
      end

      it "returns nil" do
        layer.init_weights().should be_nil
        layer.init_weights( 2.5 ).should be_nil
        layer.init_weights( -0.7, 1.0 ).should be_nil
      end
    end

    describe "#set_input" do
      it "uses sfloat parameter as new input attribute directly" do
        i = NArray.sfloat(3).random()
        layer.set_input i
        layer.input.should be i
      end

      it "casts other NArray types to sfloat, and uses a copy" do
        i = NArray.float(3).random()
        layer.set_input i
        layer.input.should_not be i
        layer.input.should be_narray_like i
      end

      it "refuses to attach wrong size of input" do
        i = NArray.sfloat(4).random()
        expect { layer.set_input i }.to raise_error
        layer.input.should be nil
      end

      it "disconnects connected layers" do
        lower_layer = CoNeNe::MLP::NLayer.new( 7, 3 )
        layer.attach_input_layer( lower_layer )

        i = NArray.sfloat(3).random()
        layer.set_input i
        layer.input.should be i
        layer.input_layer.should be_nil
        lower_layer.output_layer.should be_nil
      end
    end

    describe "#attach_input_layer" do
      it "uses parameter as new input_layer attribute directly" do
        il = CoNeNe::MLP::NLayer.new( 7, 3 )
        layer.attach_input_layer il
        layer.input_layer.should be il
        layer.input.should be il.output
      end

      it "refuses to attach wrong size of input" do
        il = CoNeNe::MLP::NLayer.new( 7, 4 )
        expect { layer.attach_input_layer il }.to raise_error
        layer.input_layer.should be nil
      end

      it "refuses to create cyclic connections" do
        il = CoNeNe::MLP::NLayer.new( 3, 3 )
        layer.attach_input_layer( il )

        expect { il.attach_input_layer layer }.to raise_error

        il.input_layer.should be_nil
        il.input.should be_nil
      end

      it "replaces existing input layer" do
        prev_layer = CoNeNe::MLP::NLayer.new( 7, 3 )
        layer.attach_input_layer( prev_layer )

        il = CoNeNe::MLP::NLayer.new( 3, 3 )
        layer.attach_input_layer( il )

        layer.attach_input_layer il
        layer.input_layer.should be il
        layer.input.should be il.output
        prev_layer.output_layer.should be_nil
      end
    end


    describe "#attach_output_layer" do
      it "uses parameter as new output_layer attribute directly" do
        ol = CoNeNe::MLP::NLayer.new( 2, 1 )
        layer.attach_output_layer ol
        layer.output_layer.should be ol
        ol.input.should be layer.output
      end

      it "refuses to attach wrong size of output" do
        ol = CoNeNe::MLP::NLayer.new( 4, 1 )
        expect { layer.attach_output_layer ol }.to raise_error
        layer.output_layer.should be nil
      end

      it "refuses to create cyclic connections" do
        ol = CoNeNe::MLP::NLayer.new( 2, 2 )
        layer.attach_output_layer( ol )

        expect { ol.attach_output_layer layer }.to raise_error

        ol.output_layer.should be_nil
        layer.input.should be_nil
      end

      it "replaces existing output connection" do
        next_layer = CoNeNe::MLP::NLayer.new( 2, 2 )
        layer.attach_output_layer( next_layer )

        ol = CoNeNe::MLP::NLayer.new( 2, 3 )
        layer.attach_output_layer( ol )

        layer.attach_output_layer ol
        layer.output_layer.should be ol
        ol.input.should be layer.output
        next_layer.input_layer.should be_nil
      end
    end

    describe "#run" do
      before :each do
        layer.set_input NArray.cast( [0.1, 0.2, 0.3], 'sfloat' )
      end

      it "calculates output associated with input and weights" do
        layer.run
        layer.output.should be_narray_like NArray[ 0.742691, 0.68568 ]
      end

      it "gives different output for different input" do
        layer.run
        result_one = layer.output.clone

        layer.set_input NArray.cast( [0.5, 0.4, 0.3], 'sfloat' )

        layer.run
        result_two = layer.output.clone

        result_one.should_not eq result_two
        result_one.should_not be_narray_like result_two
      end

      it "gives similar output for similar input" do
        layer.run
        result_one = layer.output.clone

        layer.set_input NArray.cast( [0.1002, 0.1998, 0.3001], 'sfloat' )

        layer.run
        result_two = layer.output.clone

        result_one.should be_narray_like result_two
        result_one.should_not eq result_two
      end

      it "sets all output values between 0 and 1" do
        layer.run
        layer.output.each do |r|
          r.should be >= 0.0
          r.should be <= 1.0
        end
      end
    end

    describe "#ms_error" do
      before :each do
        layer.set_input NArray.cast( [0.1, 0.4, 0.9], 'sfloat' )
        layer.run
      end

      it "returns current error value for network" do
        err = layer.ms_error( NArray.cast( [0.5, 1.0], 'sfloat' ) )
        err.should be_within(1e-6).of 0.089057

        err = layer.ms_error( NArray.cast( [0.0, 0.5], 'sfloat' ) )
        err.should be_within(1e-6).of 0.390664
      end

      it "returns similar values for similar targets" do
        err = layer.ms_error( NArray.cast( [0.12, 0.12], 'sfloat' ) )
        err.should be_within(1e-6).of 0.466518

        err = layer.ms_error( NArray.cast( [0.13, 0.11], 'sfloat' ) )
        err.should be_within(1e-6).of 0.465739
      end
    end

    describe "#calc_output_deltas" do
      before :each do
        layer.set_input NArray.cast( [0.1, 0.4, 0.9 ], 'sfloat' )
        layer.run
      end

      it "returns an array of error values" do
        errs = layer.calc_output_deltas( NArray.cast( [0.5, 1.0], 'sfloat' ) )
        errs.should be_narray_like NArray[ 0.0451288, -0.0444903 ]
      end

      it "sets output_deltas" do
        layer.calc_output_deltas( NArray.cast( [0.5, 1.0], 'sfloat' ) )
        layer.output_deltas.should be_narray_like NArray[ 0.0451288, -0.0444903 ]
      end
    end

    # TODO: Confirm these specific values with Ruby version
    describe "#backprop_deltas" do
      before :each do
        weights = NArray.cast( [ [ -0.1, 0.5, -0.2 ] ], 'sfloat' )
        @ol = CoNeNe::MLP::NLayer.from_weights( weights )

        layer.set_input NArray.cast( [0.1, 0.4, 0.9 ], 'sfloat' )
        layer.attach_output_layer( @ol )
        layer.run
        @ol.run
        @ol.calc_output_deltas( NArray.cast( [1.0], 'sfloat' ) )
      end

      it "returns an array of delta values" do
        deltas = @ol.backprop_deltas
        deltas.should be_narray_like NArray[ 0.00155221, -0.0109102 ]
      end

      it "back propagates the deltas to input layer" do
        @ol.backprop_deltas
        layer.output_deltas.should be_narray_like NArray[ 0.00155221, -0.0109102 ]
      end
    end

  end
end
