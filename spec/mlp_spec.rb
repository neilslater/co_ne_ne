require 'helpers'

describe CoNeNe::MLP do
  describe "helper methods" do
    describe "#transfer" do
      it "is monotonically increasing function value between 0.0 and 1.0" do
        results = [-120.0, -12.0, -1.2, -0.4, 0.0, 0.01, 0.23, 4.5, 7.8 ].map do |x|
          CoNeNe::MLP.transfer( x )
        end
        results.all? { |r| r >= 0.0 && r <= 1.0 }.should be_true
        results.sort.should == results
        results.first.should be_within(0.001).of 0.0
        results.last.should be_within(0.001).of 1.0
      end
    end

    describe "#transfer_derivative" do
      it "is always positive, with lowest values at extremes" do
        results = [-120.0, -12.0, -1.2, -0.4, 0.0, 0.01, 0.23, 4.5, 7.8 ].map do |x|
          CoNeNe::MLP.transfer_derivative( CoNeNe::MLP.transfer( x ) )
        end
        results.all? { |r| r > 0.0 }.should be_true
        results.first.should be_within(0.001).of 0.0
        results.sort.last.should be > 0.2
        results.last.should be_within(0.001).of 0.0
      end
    end
  end
end

describe CoNeNe::MLP::Layer do
  describe "class methods" do
    describe "#new" do
      it "creates a new layer" do
        CoNeNe::MLP::Layer.new( 2, 1 ).should be_a CoNeNe::MLP::Layer
      end

      it "initialises weights, and output arrays" do
        layer = CoNeNe::MLP::Layer.new( 4, 3 )
        layer.weights.should be_a NArray
        layer.weights.shape.should == [ 5, 3 ]

        layer.output.should be_a NArray
        layer.output.shape.should == [ 3 ]
      end
    end

    describe "#from_weights" do
      it "creates a new layer" do
        CoNeNe::MLP::Layer.from_weights( NArray.sfloat(4,5) ).should be_a CoNeNe::MLP::Layer
      end

      it "initialises sizes and output arrays" do
        layer = CoNeNe::MLP::Layer.from_weights( NArray.sfloat(4,5) )

        layer.num_inputs.should be 3
        layer.num_outputs.should be 5

        layer.output.should be_a NArray
        layer.output.shape.should == [ 5 ]
      end

      it "assigns to the weights attribute directly (not a copy)" do
        w =  NArray.sfloat(12,7)
        layer = CoNeNe::MLP::Layer.from_weights( w )
        layer.weights.should be w
      end
    end

  end

  describe "instance methods" do
    let :layer do
      weights = NArray.cast(
        [ [ -0.1, 0.5, 0.9, 0.3, 0.7 ], [ -0.6, 0.6, 0.4, -0.3, 0.6 ],
          [ 0.9, -0.1, 0.3, -0.8, -0.8 ] ], 'sfloat' )
      CoNeNe::MLP::Layer.from_weights( weights )
    end

    describe "#attach_input" do
      it "uses parameter as new input attribute directly" do
        i = NArray.sfloat(4).random()
        layer.attach_input i
        layer.input.should be i
      end

      it "refuses to attach wrong size of input" do
        i = NArray.sfloat(3).random()
        expect { layer.attach_input i }.to raise_error
        layer.input.should be nil
      end

      it "disconnects connected layers" do
        lower_layer = CoNeNe::MLP::Layer.new( 7, 4 )
        layer.attach_input_layer( lower_layer )

        i = NArray.sfloat(4).random()
        layer.attach_input i
        layer.input.should be i
        layer.input_layer.should be_nil
        lower_layer.output_layer.should be_nil
      end
    end

    describe "#attach_input_layer" do
      it "uses parameter as new input_layer attribute directly" do
        il = CoNeNe::MLP::Layer.new( 7, 4 )
        layer.attach_input_layer il
        layer.input_layer.should be il
        layer.input.should be il.output
      end

      it "refuses to attach wrong size of input" do
        il = CoNeNe::MLP::Layer.new( 7, 3 )
        expect { layer.attach_input_layer il }.to raise_error
        layer.input_layer.should be nil
      end

      it "refuses to create cyclic connections" do
        il = CoNeNe::MLP::Layer.new( 3, 4 )
        layer.attach_input_layer( il )

        expect { il.attach_input_layer layer }.to raise_error

        il.input_layer.should be_nil
        il.input.should be_nil
      end

      it "replaces existing input connection" do
        prev_layer = CoNeNe::MLP::Layer.new( 7, 4 )
        layer.attach_input_layer( prev_layer )

        il = CoNeNe::MLP::Layer.new( 3, 4 )
        layer.attach_input_layer( il )

        layer.attach_input_layer il
        layer.input_layer.should be il
        layer.input.should be il.output
        prev_layer.output_layer.should be_nil
      end
    end

    describe "#attach_output_layer" do
      it "uses parameter as new output_layer attribute directly" do
        ol = CoNeNe::MLP::Layer.new( 3, 1 )
        layer.attach_output_layer ol
        layer.output_layer.should be ol
        ol.input.should be layer.output
      end

      it "refuses to attach wrong size of output" do
        ol = CoNeNe::MLP::Layer.new( 4, 1 )
        expect { layer.attach_output_layer ol }.to raise_error
        layer.output_layer.should be nil
      end

      it "refuses to create cyclic connections" do
        ol = CoNeNe::MLP::Layer.new( 3, 2 )
        layer.attach_output_layer( ol )

        expect { ol.attach_output_layer layer }.to raise_error

        ol.output_layer.should be_nil
        layer.input.should be_nil
      end

      it "replaces existing output connection" do
        next_layer = CoNeNe::MLP::Layer.new( 3, 2 )
        layer.attach_output_layer( next_layer )

        ol = CoNeNe::MLP::Layer.new( 3, 3 )
        layer.attach_output_layer( ol )

        layer.attach_output_layer ol
        layer.output_layer.should be ol
        ol.input.should be layer.output
        next_layer.input_layer.should be_nil
      end
    end

    describe "#run" do
      before :each do
        layer.attach_input NArray.cast( [0.1, 0.2, 0.3, 0.4], 'sfloat' )
      end

      it "modifies output" do
        layer.output.should eq NArray.sfloat(3)
        layer.run
        layer.output.should_not eq NArray.sfloat(3)
      end

      it "calculates output associated with input and weights" do
        layer.run
        layer.output.should be_narray_like NArray[ 0.764948, 0.65926, 0.276878 ]
      end

      it "gives different output for different input" do
        layer.run
        result_one = layer.output.clone

        layer.attach_input NArray.cast( [0.5, 0.4, 0.3, 0.2], 'sfloat' )

        layer.run
        result_two = layer.output.clone

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

    describe "#rms_error" do
      before :each do
        layer.attach_input NArray.cast( [0.1, 0.4, 0.9, 0.2], 'sfloat' )
        layer.run
      end

      it "returns current error value for network" do
        err = layer.rms_error( NArray.cast( [0.5, 1.0, 0.0], 'sfloat' ) )
        err.should be_within(1e-6).of 0.102739

        err = layer.rms_error( NArray.cast( [0.0, 0.5, 1.0], 'sfloat' ) )
        err.should be_within(1e-6).of 0.405809
      end

    end

    describe "#calc_output_deltas" do
      before :each do
        layer.attach_input NArray.cast( [0.1, 0.4, 0.9, 0.2], 'sfloat' )
        layer.run
      end

      it "returns an array of error values" do
        errs = layer.calc_output_deltas( NArray.cast( [0.5, 1.0, 0.0], 'sfloat' ) )
        errs.should be_narray_like NArray[ -0.044237, 0.0479737, -0.0780434 ]
      end

      it "sets output_deltas" do
        layer.calc_output_deltas( NArray.cast( [0.5, 1.0, 0.0], 'sfloat' ) )
        layer.output_deltas.should be_narray_like NArray[ -0.044237, 0.0479737, -0.0780434 ]
      end
    end

    describe "#backprop_deltas" do
      before :each do
        weights = NArray.cast( [ [ -0.1, 0.5, 0.9, 0.3 ], [ -0.6, 0.6, 0.4, -0.3 ] ], 'sfloat' )
        @ol = CoNeNe::MLP::Layer.from_weights( weights )

        layer.attach_input NArray.cast( [0.1, 0.4, 0.9, 0.2], 'sfloat' )
        layer.attach_output_layer( @ol )
        layer.run
        @ol.run
        @ol.calc_output_deltas( NArray.cast( [0.5, 1.0], 'sfloat' ) )
      end

      it "returns an array of delta values" do
        deltas = @ol.backprop_deltas
        deltas.should be_narray_like NArray[ -0.00977509, 0.0114911, 0.00360203 ]
      end

      it "back propagates the deltas to input layer" do
        @ol.backprop_deltas
        layer.output_deltas.should be_narray_like NArray[ -0.00977509, 0.0114911, 0.00360203 ]
      end
    end

    describe "#update_weights" do
      before :each do
        layer.attach_input NArray.cast( [0.1, 0.4, 0.9, 0.2], 'sfloat' )
        layer.run
        layer.calc_output_deltas( NArray.cast( [0.5, 1.0, 0.0], 'sfloat' ) )
      end

      it "alters the weights" do
        layer.update_weights( 1.0 )
        original_weights = NArray.cast(
          [ [ -0.1, 0.5, 0.9, 0.3, 0.7 ], [ -0.6, 0.6, 0.4, -0.3, 0.6 ],
            [ 0.9, -0.1, 0.3, -0.8, -0.8 ] ], 'sfloat' )
        layer.weights.should_not be_narray_like original_weights
        diff = original_weights - layer.weights
        (diff * diff).sum.should be > 0.01
      end

      it "reduces the rms error" do
        target = NArray.cast( [0.0, 0.5, 1.0], 'sfloat' )
        original_err =  layer.rms_error( target )
        10.times do
          layer.run
          layer.calc_output_deltas( target )
          layer.update_weights( 1.0 )
        end
        new_err = layer.rms_error( target )
        (original_err - new_err).should be > 0.1
      end
    end


  end
end

describe CoNeNe::MLP::Network do
  describe "class methods" do
    describe "#new" do
      it "creates a new network" do
        CoNeNe::MLP::Network.new( 2, [4], 1 ).should be_a CoNeNe::MLP::Network
      end

      it "creates layers" do
        nn = CoNeNe::MLP::Network.new( 2, [4], 1 )
        nn.layers.should be_a Array
        nn.layers.count.should == 2

        nn.layers.first.should be_a CoNeNe::MLP::Layer
        nn.layers.first.num_inputs.should == 2
        nn.layers.first.num_outputs.should == 4

        nn.layers.last.should be_a CoNeNe::MLP::Layer
        nn.layers.last.num_inputs.should == 4
        nn.layers.last.num_outputs.should == 1
      end
    end
  end

  describe "instance methods" do
    let( :nn ) { CoNeNe::MLP::Network.new( 2, [4], 1 ) }
    let( :nn2 ) { CoNeNe::MLP::Network.new( 2, [5,3], 1 ) }
    let( :nn3 ) { CoNeNe::MLP::Network.new( 2, [4,3,2], 1 ) }
    let( :xor_train_set ) {
      [
        [  NArray.cast( [0.0, 0.0], 'sfloat' ), NArray.cast( [0.0], 'sfloat' ) ],
        [  NArray.cast( [0.0, 1.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) ],
        [  NArray.cast( [1.0, 0.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) ],
        [  NArray.cast( [1.0, 1.0], 'sfloat' ), NArray.cast( [0.0], 'sfloat' ) ]
      ]
    }

    describe "#run" do
      it "modifies output" do
        nn.output.should eq NArray.sfloat(1)
        nn.run( NArray.cast( [1.0, 0.0], 'sfloat' ) )
        nn.output.should_not eq NArray.sfloat(1)
      end
    end

    describe "#train_once" do
      it "modifies output" do
        nn.train_once( NArray.cast( [1.0, 0.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) )
        nn.output.should_not eq NArray.sfloat(1)
      end

      it "can learn xor when run repeatedly" do
        rms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn.run xin
          rms_total += nn.rms_error( xtarg )
        end
        rms_total /= 4

        3000.times do
          xor_train_set.each do | xin, xtarg |
            nn.train_once xin, xtarg
          end
        end

        after_rms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn.run xin
          after_rms_total += nn.rms_error( xtarg )
        end
        after_rms_total /= 4

        after_rms_total.should be < rms_total

        nn.run( NArray.cast( [0.0, 0.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
        nn.run( NArray.cast( [0.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn.run( NArray.cast( [1.0, 0.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn.run( NArray.cast( [1.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
      end

      it "can learn xor with 2 hidden layers" do
        rms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn2.run xin
          rms_total += nn2.rms_error( xtarg )
        end
        rms_total /= 4

        10000.times do
          xor_train_set.each do | xin, xtarg |
            nn2.train_once xin, xtarg
          end
        end

        after_rms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn2.run xin
          after_rms_total += nn2.rms_error( xtarg )
        end
        after_rms_total /= 4

        after_rms_total.should be < rms_total

        nn2.run( NArray.cast( [0.0, 0.0], 'sfloat' ) )[0].should be_within(0.15).of 0.0
        nn2.run( NArray.cast( [0.0, 1.0], 'sfloat' ) )[0].should be_within(0.15).of 1.0
        nn2.run( NArray.cast( [1.0, 0.0], 'sfloat' ) )[0].should be_within(0.15).of 1.0
        nn2.run( NArray.cast( [1.0, 1.0], 'sfloat' ) )[0].should be_within(0.15).of 0.0
      end
    end
  end
end
