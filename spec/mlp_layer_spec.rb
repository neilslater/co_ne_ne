require 'helpers'

describe CoNeNe::MLP::Layer do
  describe "class methods" do
    describe "#new" do
      it "creates a new layer" do
        expect( CoNeNe::MLP::Layer.new( 2, 1 ) ).to be_a CoNeNe::MLP::Layer
      end

      it "refuses to create new layers for bad parameters" do
        expect { CoNeNe::MLP::Layer.new( 0, 2 ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( 3, -1 ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( "hello", 2 ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( 3, 2, "garbage" ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( 3, 2, :foobar ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( 3, 2, :tanh, 17 ) }.to raise_error
      end

      it "sets values of attributes based on input and output size" do
        CoNeNe.srand( 7000 )

        layer = CoNeNe::MLP::Layer.new( 3, 2 )
        expect( layer.num_inputs ).to be 3
        expect( layer.num_outputs ).to be 2
        expect( layer.transfer ).to be CoNeNe::Transfer::Sigmoid

        expect( layer.weights ).to be_narray_like NArray[
            [ 0.088395, 0.759151, -0.00383174, 0.756848 ],
            [ -0.422237, -0.552743, -0.128862, 0.774815 ] ]

        expect( layer.weights_last_deltas ).to be_a NArray
        expect( layer.weights_last_deltas.shape ).to eql [4,2]

        expect( layer.output ).to be_a NArray
        expect( layer.output.shape ).to eql [2]

        expect( layer.output_deltas ).to be_a NArray
        expect( layer.output_deltas.shape ).to eql [2]
      end

      it "accepts an optional transfer function type param" do
        layer = CoNeNe::MLP::Layer.new( 4, 1, :sigmoid )
        expect( layer.transfer ).to be CoNeNe::Transfer::Sigmoid

        layer = CoNeNe::MLP::Layer.new( 5, 3, :tanh )
        expect( layer.transfer ).to be CoNeNe::Transfer::TanH

        layer = CoNeNe::MLP::Layer.new( 7, 2, :relu )
        expect( layer.transfer ).to be CoNeNe::Transfer::ReLU
      end
    end

    describe "#from_weights" do
      it "creates a new layer" do
        expect( CoNeNe::MLP::Layer.from_weights( NArray.sfloat(4,5) ) ).to be_a CoNeNe::MLP::Layer
      end

      it "initialises sizes and output arrays" do
        layer = CoNeNe::MLP::Layer.from_weights( NArray.sfloat(4,5) )

        expect( layer.num_inputs ).to be 3
        expect( layer.num_outputs ).to be 5

        expect( layer.output ).to be_a NArray
        expect( layer.output.shape ).to eql [ 5 ]
      end

      it "assigns to the weights attribute directly (not a copy)" do
        w =  NArray.sfloat(12,7)
        layer = CoNeNe::MLP::Layer.from_weights( w )
        expect( layer.weights ).to be w
      end

      it "accepts an optional transfer function type param" do
        w =  NArray.sfloat(3,2)
        layer = CoNeNe::MLP::Layer.from_weights( w, :sigmoid )
        expect( layer.transfer ).to be CoNeNe::Transfer::Sigmoid

        w =  NArray.sfloat(3,2)
        layer = CoNeNe::MLP::Layer.from_weights( w, :tanh )
        expect( layer.transfer ).to be CoNeNe::Transfer::TanH

        w =  NArray.sfloat(3,2)
        layer = CoNeNe::MLP::Layer.from_weights( w, :relu )
        expect( layer.transfer ).to be CoNeNe::Transfer::ReLU
      end

      it "refuses to create new layers for bad parameters" do
        expect { CoNeNe::MLP::Layer.new( NArray.sfloat(3,2,1) ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( NArray.sfloat(1,2) ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( NArray.sfloat(7)) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( NArray.sfloat(5,2), "NOTVALID" ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( NArray.sfloat(4,1), :blah ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( NArray.sfloat(4,1), :tanh, "extras" ) }.to raise_error
      end
    end

    describe "with Marshal" do
      it "can save and retrieve a layer, preserving weights and transfer function" do
        orig_layer = CoNeNe::MLP::Layer.new( 5, 3, :sigmoid )
        saved = Marshal.dump( orig_layer )
        copy_layer = Marshal.load( saved )

        expect( copy_layer ).to_not be orig_layer
        expect( copy_layer.num_inputs ).to be 5
        expect( copy_layer.num_outputs ).to be 3
        expect( copy_layer.transfer ).to be CoNeNe::Transfer::Sigmoid
        expect( copy_layer.weights ).to be_narray_like orig_layer.weights
      end
    end

  end

  describe "instance methods" do
    let :layer do
      weights = NArray.cast( [ [ -0.1, 0.5, 0.9, 0.7 ], [ -0.6, 0.6, 0.4, 0.6 ] ], 'sfloat' )
      CoNeNe::MLP::Layer.from_weights( weights )
    end

    describe "#clone" do
      it "should make a simple copy of number of inputs, outputs and transfer function" do
        copy = layer.clone
        expect( copy.num_inputs ).to eql layer.num_inputs
        expect( copy.num_outputs ).to eql layer.num_outputs
        expect( copy.transfer ).to eql layer.transfer
      end

      it "should deep clone all arrays of weights and output" do
        copy = layer.clone

        expect( copy.weights ).to_not be layer.weights
        expect( copy.weights ).to be_narray_like layer.weights

        expect( copy.output ).to_not be layer.output
        expect( copy.output ).to be_narray_like layer.output

        expect( copy.weights ).to_not be layer.weights
        expect( copy.weights ).to be_narray_like layer.weights

        expect( copy.output_deltas ).to_not be layer.output_deltas
        expect( copy.output_deltas ).to be_narray_like layer.output_deltas

        expect( copy.weights_last_deltas ).to_not be layer.weights_last_deltas
        expect( copy.weights_last_deltas ).to be_narray_like layer.weights_last_deltas
      end

      it "should disconnect inputs and outputs" do
        layer.attach_input_layer CoNeNe::MLP::Layer.new( 4, 3 )
        layer.attach_output_layer CoNeNe::MLP::Layer.new( 2, 1 )
        copy = layer.clone

        expect( copy.input ).to be_nil
        expect( copy.input_layer ).to be_nil
        expect( copy.output_layer ).to be_nil
      end

      it "should copy the transfer function" do
        layer2 = CoNeNe::MLP::Layer.new( 4, 3, :tanh )
        copy = layer2.clone
        expect( copy.transfer ).to be CoNeNe::Transfer::TanH

        layer3 = CoNeNe::MLP::Layer.new( 4, 3, :relu )
        copy = layer3.clone
        expect( copy.transfer ).to be CoNeNe::Transfer::ReLU
      end
    end

    describe "#init_weights" do
      before :each do
        CoNeNe.srand(800)
      end

      it "should set weights in range -0.8 to 0.8 by default" do
        layer.init_weights
        expect( layer.weights ).to be_narray_like NArray[
            [ 0.130206, 0.520598, 0.614333, -0.275051 ],
            [ 0.564844, -0.236939, 0.256392, -0.466942 ] ]
      end

      it "should accept a single param to set +- range" do
        layer.init_weights( 4.0 )
        expect( layer.weights ).to be_narray_like NArray[
            [ 0.65103, 2.60299, 3.07167, -1.37525 ],
            [ 2.82422, -1.18469, 1.28196, -2.33471 ] ]
      end

      it "should accept two params to select from a range" do
        layer.init_weights( 0.2, 1.8 )
        expect( layer.weights ).to be_narray_like NArray[
            [ 1.13021, 1.5206, 1.61433, 0.724949 ],
            [ 1.56484, 0.763061, 1.25639, 0.533058 ] ]
      end

      it "should work with a negative single param" do
        layer.init_weights( -0.8 )
        expect( layer.weights ).to be_narray_like NArray[
            [ -0.130206, -0.520598, -0.614333, 0.275051 ],
            [ -0.564844, 0.236939, -0.256392, 0.466942 ] ]
      end

      it "should work with a 'reversed' range" do
        layer.init_weights( 1.0, 0.0 )
        expect( layer.weights ).to be_narray_like NArray[
            [ 0.418621, 0.174627, 0.116042, 0.671907 ],
            [ 0.146972, 0.648087, 0.339755, 0.791839 ] ]
      end

      it "should raise an error for non-numeric params" do
        expect { layer.init_weights( [] ) }.to raise_error
        expect { layer.init_weights( "Hi" ) }.to raise_error
        expect { layer.init_weights( 2.5, :foo => 'bar' ) }.to raise_error
      end

      it "returns nil" do
        expect( layer.init_weights() ).to be_nil
        expect( layer.init_weights( 2.5 ) ).to be_nil
        expect( layer.init_weights( -0.7, 1.0 ) ).to be_nil
      end
    end

    describe "#set_input" do
      it "uses sfloat parameter as new input attribute directly" do
        i = NArray.sfloat(3).random()
        layer.set_input i
        expect( layer.input ).to be i
      end

      it "casts other NArray types to sfloat, and uses a copy" do
        i = NArray.float(3).random()
        layer.set_input i
        expect( layer.input ).to_not be i
        expect( layer.input ).to be_narray_like i
      end

      it "refuses to attach wrong size of input" do
        i = NArray.sfloat(4).random()
        expect { layer.set_input i }.to raise_error
        expect( layer.input ).to be nil
      end

      it "disconnects connected layers" do
        lower_layer = CoNeNe::MLP::Layer.new( 7, 3 )
        layer.attach_input_layer( lower_layer )

        i = NArray.sfloat(3).random()
        layer.set_input i
        expect( layer.input ).to be i
        expect( layer.input_layer ).to be_nil
        expect( lower_layer.output_layer ).to be_nil
      end
    end

    describe "#attach_input_layer" do
      it "uses parameter as new input_layer attribute directly" do
        il = CoNeNe::MLP::Layer.new( 7, 3 )
        layer.attach_input_layer il
        expect( layer.input_layer ).to be il
        expect( layer.input ).to be il.output
      end

      it "refuses to attach wrong size of input" do
        il = CoNeNe::MLP::Layer.new( 7, 4 )
        expect { layer.attach_input_layer il }.to raise_error
        expect( layer.input_layer ).to be nil
      end

      it "refuses to create cyclic connections" do
        il = CoNeNe::MLP::Layer.new( 3, 3 )
        layer.attach_input_layer( il )

        expect { il.attach_input_layer layer }.to raise_error

        expect( il.input_layer ).to be_nil
        expect( il.input ).to be_nil
      end

      it "replaces existing input layer" do
        prev_layer = CoNeNe::MLP::Layer.new( 7, 3 )
        layer.attach_input_layer( prev_layer )

        il = CoNeNe::MLP::Layer.new( 3, 3 )
        layer.attach_input_layer( il )

        layer.attach_input_layer il
        expect( layer.input_layer ).to be il
        expect( layer.input ).to be il.output
        expect( prev_layer.output_layer ).to be_nil
      end
    end

    describe "#attach_output_layer" do
      it "uses parameter as new output_layer attribute directly" do
        ol = CoNeNe::MLP::Layer.new( 2, 1 )
        layer.attach_output_layer ol
        expect( layer.output_layer ).to be ol
        expect( ol.input ).to be layer.output
      end

      it "refuses to attach wrong size of output" do
        ol = CoNeNe::MLP::Layer.new( 4, 1 )
        expect { layer.attach_output_layer ol }.to raise_error
        expect( layer.output_layer ).to be nil
      end

      it "refuses to create cyclic connections" do
        ol = CoNeNe::MLP::Layer.new( 2, 2 )
        layer.attach_output_layer( ol )

        expect { ol.attach_output_layer layer }.to raise_error

        expect( ol.output_layer ).to be_nil
        expect( layer.input ).to be_nil
      end

      it "replaces existing output connection" do
        next_layer = CoNeNe::MLP::Layer.new( 2, 2 )
        layer.attach_output_layer( next_layer )

        ol = CoNeNe::MLP::Layer.new( 2, 3 )
        layer.attach_output_layer( ol )

        layer.attach_output_layer ol
        expect( layer.output_layer ).to be ol
        expect( ol.input ).to be layer.output
        expect( next_layer.input_layer ).to be_nil
      end
    end

    describe "#run" do
      before :each do
        layer.set_input NArray.cast( [0.1, 0.2, 0.3], 'sfloat' )
      end

      it "calculates output associated with input and weights" do
        layer.run
        expect( layer.output ).to be_narray_like NArray[ 0.742691, 0.68568 ]
      end

      it "gives different output for different input" do
        layer.run
        result_one = layer.output.clone

        layer.set_input NArray.cast( [0.5, 0.4, 0.3], 'sfloat' )

        layer.run
        result_two = layer.output.clone

        expect( result_one ).to_not eq result_two
        expect( result_one ).to_not be_narray_like result_two
      end

      it "gives similar output for similar input" do
        layer.run
        result_one = layer.output.clone

        layer.set_input NArray.cast( [0.1002, 0.1998, 0.3001], 'sfloat' )

        layer.run
        result_two = layer.output.clone

        expect( result_one ).to be_narray_like result_two
        expect( result_one ).to_not eq result_two
      end

      it "sets all output values between 0 and 1" do
        layer.run
        layer.output.each do |r|
          expect( r ).to be >= 0.0
          expect( r ).to be <= 1.0
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
        expect( err ).to be_within(1e-6).of 0.089057

        err = layer.ms_error( NArray.cast( [0.0, 0.5], 'sfloat' ) )
        expect( err ).to be_within(1e-6).of 0.390664
      end

      it "returns similar values for similar targets" do
        err = layer.ms_error( NArray.cast( [0.12, 0.12], 'sfloat' ) )
        expect( err ).to be_within(1e-6).of 0.466518

        err = layer.ms_error( NArray.cast( [0.13, 0.11], 'sfloat' ) )
        expect( err ).to be_within(1e-6).of 0.465739
      end
    end

    describe "#calc_output_deltas" do
      before :each do
        layer.set_input NArray.cast( [0.1, 0.4, 0.9 ], 'sfloat' )
        layer.run
      end

      it "returns an array of error values" do
        errs = layer.calc_output_deltas( NArray.cast( [0.5, 1.0], 'sfloat' ) )
        expect( errs ).to be_narray_like NArray[ -0.0451288, 0.0444903 ]
      end

      it "sets output_deltas" do
        layer.calc_output_deltas( NArray.cast( [0.5, 1.0], 'sfloat' ) )
        expect( layer.output_deltas ).to be_narray_like NArray[ -0.0451288, 0.0444903 ]
      end
    end

    describe "#backprop_deltas" do
      before :each do
        weights = NArray.cast( [ [ -0.1, 0.5, -0.2 ] ], 'sfloat' )
        @ol = CoNeNe::MLP::Layer.from_weights( weights )

        layer.set_input NArray.cast( [0.1, 0.4, 0.9 ], 'sfloat' )
        layer.attach_output_layer( @ol )
        layer.run
        @ol.run
        @ol.calc_output_deltas( NArray.cast( [1.0], 'sfloat' ) )
      end

      it "returns an array of delta values" do
        deltas = @ol.backprop_deltas
        expect( deltas ).to be_narray_like NArray[ -0.00155221, 0.0109102 ]
      end

      it "back propagates the deltas to input layer" do
        @ol.backprop_deltas
        expect( layer.output_deltas ).to be_narray_like NArray[ -0.00155221, 0.0109102 ]
      end
    end

    describe "#update_weights" do
      before :each do
        layer.set_input NArray.cast( [0.1, 0.4, 0.9], 'sfloat' )
        layer.run
        layer.calc_output_deltas( NArray.cast( [0.5, 1.0], 'sfloat' ) )
      end

      it "alters the weights" do
        layer.update_weights( 1.0 )
        original_weights = NArray.cast( [ [ -0.1, 0.5, 0.9, 0.7 ], [ -0.6, 0.6, 0.4, 0.6 ] ], 'sfloat' )
        expect( layer.weights ).to_not be_narray_like original_weights
        diff = original_weights - layer.weights
        (diff * diff).sum.should be > 0.001
      end

      it "reduces the mean square error" do
        target = NArray.cast( [1.0, 0.0], 'sfloat' )
        original_err =  layer.ms_error( target )
        50.times do
          layer.run
          layer.calc_output_deltas( target )
          layer.update_weights( 1.0 )
        end
        new_err = layer.ms_error( target )
        (original_err - new_err).should be > 0.05
      end
    end
  end
end
