require 'helpers'

describe CoNeNe::MLP::Network do
  describe "class methods" do
    describe "#new" do
      it "creates a new network" do
        CoNeNe::MLP::Network.new( 2, [], 1 ).should be_a CoNeNe::MLP::Network
        CoNeNe::MLP::Network.new( 2, [4], 1 ).should be_a CoNeNe::MLP::Network
        CoNeNe::MLP::Network.new( 2, [4,2], 1 ).should be_a CoNeNe::MLP::Network
      end

      it "does not create a new network if any params are missing or bad" do
        expect { CoNeNe::MLP::Network.new( -2, [4], 1 ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( nil, [4], 1 ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( "a fish", [4], 1 ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( [4], 1 ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( 2, 3, 1 ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( 2, ["z"], 1 ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( 2, [-3], 1 ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( 2, [3,4], -81 ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( 2, [3] ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( 2, [3], nil ) }.to raise_error
        expect { CoNeNe::MLP::Network.new( 2, [3], 'a frog' ) }.to raise_error
      end

      it "creates a network with right number of Layers" do
        network = CoNeNe::MLP::Network.new( 2, [], 1 )
        network.num_layers.should == 1
        network.layers.count.should == 1
        layer = network.layers.first
        layer.num_inputs.should == 2
        layer.num_outputs.should == 1

        network = CoNeNe::MLP::Network.new( 2, [2], 1 )
        network.num_layers.should == 2
        layer = network.layers.first
        layer.num_inputs.should == 2
        layer.num_outputs.should == 2
        layer = network.layers.last
        layer.num_inputs.should == 2
        layer.num_outputs.should == 1

        network = CoNeNe::MLP::Network.new( 2, [5,6,4,2], 1 )
        network.num_layers.should == 5
        layers = network.layers
        [ [2,5], [5,6], [6,4], [4,2], [2,1] ].each do |xp_in, xp_out|
          layer = layers.shift
          layer.num_inputs.should == xp_in
          layer.num_outputs.should == xp_out
        end
      end

      it "creates a network with right number of inputs and outputs" do
        network = CoNeNe::MLP::Network.new( 2, [], 1 )
        network.num_inputs.should == 2
        network.input.should be_nil
        network.num_outputs.should == 1
        network.output.should be_a NArray
        network.output.shape.should == [1]

        network = CoNeNe::MLP::Network.new( 2, [7,3,2], 2 )
        network.num_inputs.should == 2
        network.input.should be_nil
        network.num_outputs.should == 2
        network.output.should be_a NArray
        network.output.shape.should == [2]
      end
    end

    describe "#from_layer" do
      let( :layer ) { CoNeNe::MLP::Layer.new( 5, 3, :tanh ) }

      it "creates a new network" do
        CoNeNe::MLP::Network.from_layer( layer ).should be_a CoNeNe::MLP::Network
      end

      it "uses layer 'as-is'" do
        network = CoNeNe::MLP::Network.from_layer( layer )
        network.layers[0].should be layer
      end

      it "uses a layer with an attached output layer" do
        layer.attach_output_layer( CoNeNe::MLP::Layer.new( 3, 2, :sigmoid ) )
        network = CoNeNe::MLP::Network.from_layer( layer )
        network.num_layers.should == 2
        network.num_outputs.should == 2
      end

      it "refuses to use a layer with an attached input layer" do
        layer.attach_input_layer( CoNeNe::MLP::Layer.new( 7, 5, :tanh ) )
        expect { CoNeNe::MLP::Network.from_layer( layer ) }.to raise_error
      end

      it "locks first layer inputs" do
        network = CoNeNe::MLP::Network.from_layer( layer )
        expect { layer.attach_input_layer( CoNeNe::MLP::Layer.new( 7, 5, :tanh ) ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( 7, 5, :tanh ).attach_output_layer( layer ) }.to raise_error
        layer.attach_output_layer( CoNeNe::MLP::Layer.new( 3, 2, :sigmoid ) )
        network.num_layers.should == 2
        network.num_outputs.should == 2
      end
    end

    describe "#from_layers" do
      let( :layers ) {  [ CoNeNe::MLP::Layer.new( 5, 3, :tanh ),
              CoNeNe::MLP::Layer.new( 3, 2, :sigmoid ) ] }

      it "creates a new network" do
        CoNeNe::MLP::Network.from_layers( layers ).should be_a CoNeNe::MLP::Network
      end

      it "uses layer objects directly" do
        network = CoNeNe::MLP::Network.from_layers( layers )
        network.layers[0].should be layers[0]
        network.layers[1].should be layers[1]
        network.num_layers.should == 2
        network.num_outputs.should == 2
      end

      it "re-assigns input" do
        layers[0].attach_input_layer( CoNeNe::MLP::Layer.new( 7, 5, :tanh ) )
        network = CoNeNe::MLP::Network.from_layers( layers )
        network.layers[0].should be layers[0]
        network.layers[1].should be layers[1]
        network.num_layers.should == 2
        network.num_outputs.should == 2
      end

      it "locks first layer inputs" do
        network = CoNeNe::MLP::Network.from_layers( layers )
        expect { layers[0].attach_input_layer( CoNeNe::MLP::Layer.new( 7, 5, :tanh ) ) }.to raise_error
        expect { CoNeNe::MLP::Layer.new( 7, 5, :tanh ).attach_output_layer( layers[0] ) }.to raise_error
        layers[1].attach_output_layer( CoNeNe::MLP::Layer.new( 2, 1, :sigmoid ) )
        network.num_layers.should == 3
        network.num_outputs.should == 1
      end
    end

    describe "with Marshal" do
      before do
        @orig_network = CoNeNe::MLP::Network.new( 2, [4], 1 )
        @saved_network = Marshal.dump( @orig_network )
        @copy_network =  Marshal.load( @saved_network )
      end

      it "can save and retrieve a network, preserving weights" do
        @copy_network.should_not be @orig_network
        @copy_network.num_inputs.should == 2
        @copy_network.num_outputs.should == 1
        @copy_network.num_layers.should == 2
        orig_layers = @orig_network.layers
        copy_layers = @copy_network.layers
        copy_layers[0].weights.should_not be orig_layers[0].weights
        copy_layers[0].weights.should be_narray_like orig_layers[0].weights
        copy_layers[1].weights.should_not be orig_layers[1].weights
        copy_layers[1].weights.should be_narray_like orig_layers[1].weights
      end

      it "restores behaviour of the network" do
        inputs =  [ NArray.cast( [1.0, 0.0], 'sfloat' ), NArray.cast( [-1.0, -1.0], 'sfloat' ),
                    NArray.cast( [0.4, 0.9], 'sfloat' ), NArray.cast( [-0.7, -0.9], 'sfloat' ),
                    NArray.cast( [0.5, 0.5], 'sfloat' ), NArray.cast( [-0.5, -0.6], 'sfloat' ),
                    NArray.cast( [0.0, 1.0], 'sfloat' ), NArray.cast( [-1.3, -1.4], 'sfloat' ) ]

        inputs.each do |i|
          r1 = @orig_network.run( i )
          r2 = @copy_network.run( i )
          r1.should_not be r2
          r1.should be_narray_like r2
        end
      end
    end
  end

  describe "instance methods" do
    let( :nn ) { CoNeNe::MLP::Network.new( 2, [4], 1 ) }
    let( :nn2 ) { CoNeNe::MLP::Network.new( 2, [5,3], 1 ) }
    let( :nn3 ) { CoNeNe::MLP::Network.new( 2, [4,3,2], 1 ) }
    let( :xor_train_set ) {
      [
        [  NArray.cast( [-1.0, -1.0], 'sfloat' ), NArray.cast( [0.0], 'sfloat' ) ],
        [  NArray.cast( [-1.0, 1.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) ],
        [  NArray.cast( [1.0, -1.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) ],
        [  NArray.cast( [1.0, 1.0], 'sfloat' ), NArray.cast( [0.0], 'sfloat' ) ]
      ]
    }

    describe "#clone" do
      it "makes a deep copy of all connected layers" do
        nn_copy = nn.clone
        nn_copy.should_not be nn
        nn_copy.num_inputs.should == 2
        nn_copy.num_outputs.should == 1
        nn_copy.num_layers.should == 2
        orig_layers = nn.layers
        copy_layers = nn_copy.layers
        copy_layers[0].weights.should_not be orig_layers[0].weights
        copy_layers[0].weights.should be_narray_like orig_layers[0].weights
        copy_layers[1].weights.should_not be orig_layers[1].weights
        copy_layers[1].weights.should be_narray_like orig_layers[1].weights
      end

      it "clones run behaviour of the network" do
        inputs =  [ NArray.cast( [1.0, 0.0], 'sfloat' ), NArray.cast( [-1.0, -1.0], 'sfloat' ),
                    NArray.cast( [0.4, 0.9], 'sfloat' ), NArray.cast( [-0.7, -0.9], 'sfloat' ),
                    NArray.cast( [0.5, 0.5], 'sfloat' ), NArray.cast( [-0.5, -0.6], 'sfloat' ),
                    NArray.cast( [0.0, 1.0], 'sfloat' ), NArray.cast( [-1.3, -1.4], 'sfloat' ) ]
        nn_copy = nn.clone

        inputs.each do |i|
          r1 = nn.run( i )
          r2 = nn_copy.run( i )
          r1.should_not be r2
          r1.should be_narray_like r2
        end
      end
    end

    describe "#init_weights" do
      before :each do
        CoNeNe.srand(900)
      end

      it "generates new values for weights" do
        layers = nn.layers
        old_weights0 = layers[0].weights.clone
        old_weights1 = layers[1].weights.clone

        nn.init_weights

        layers[0].weights.should_not be_narray_like old_weights0
        layers[1].weights.should_not be_narray_like old_weights1

        layers[0].weights.should be_narray_like NArray[ [ 0.29383, 0.33766, 0.747574 ],
              [ -0.59579, -0.590868, -0.436368 ],
              [ 0.528037, 0.62401, -0.44007 ],
              [ 0.679676, -0.0280843, 0.212935 ] ]

        layers[1].weights.should be_narray_like NArray[ [ 0.737034, 0.514718, -0.50875, -0.129045, -0.46841 ] ]
      end

      it "takes optional scaling params, affecting all layers" do
        layers = nn.layers

        nn.init_weights( 1.6 )
        layers = nn.layers

        layers[0].weights.should be_narray_like NArray[ [ 0.58766, 0.675321, 1.49515 ],
              [ -1.19158, -1.18174, -0.872735 ],
              [ 1.05607, 1.24802, -0.88014 ],
              [ 1.35935, -0.0561687, 0.42587 ] ]

        layers[1].weights.should be_narray_like NArray[ [ 1.47407, 1.02944, -1.0175, -0.25809, -0.936819 ] ]

        nn.init_weights( 1.0, 3.0 )

        layers[0].weights.should be_narray_like NArray[ [ 2.03394, 2.44456, 1.62576 ],
              [ 2.78554, 1.73815, 1.91262 ],
              [ 1.51784, 1.61086, 2.34845 ],
              [ 1.42056, 1.55768, 2.44562 ] ]

        layers[1].weights.should be_narray_like NArray[ [ 2.55728, 2.45761, 2.34661, 2.9218, 2.18352 ] ]
      end
    end

    describe "#run" do
      it "modifies output" do
        nn.run( NArray.cast( [1.0, 0.0], 'sfloat' ) )
        nn.output.should_not eq NArray.sfloat(1)
      end

      it "returns the output object" do
        result = nn.run( NArray.cast( [1.0, 0.0], 'sfloat' ) )
        nn.output.should eq result
        nn.output.should be result
      end

      it "alters output of each layer" do
        CoNeNe.srand(900)

        nn.init_weights
        nn.run( NArray.cast( [1.0, 0.0], 'sfloat' ) )
        layers = nn.layers

        layers[0].output.should be_narray_like NArray[ 0.778442, -0.774772, 0.0877405, 0.712681 ]
        layers[1].output.should be_narray_like NArray[ 0.39411 ]

        nn2.init_weights
        nn2.run( NArray.cast( [0.3, -0.4], 'sfloat' ) )
        layers = nn2.layers

        layers[0].output.should be_narray_like NArray[ -0.217227, -0.120661, -0.153031, -0.130874, 0.622694 ]
        layers[1].output.should be_narray_like NArray[ -0.183053, 0.528921, -0.806468 ]
        layers[2].output.should be_narray_like NArray[ 0.585357 ]
      end
    end

    describe "#ms_error" do
      it "should calculate mean square error of current output compared to target" do
        CoNeNe.srand(900)
        nn.init_weights
        nn.run( NArray.cast( [1.0, 0.0], 'sfloat' ) )
        nn.ms_error( NArray.cast( [1.0], 'sfloat' ) ).should be_within(1e-6).of 0.367103
      end
    end

    describe "#train_once" do
      it "returns ms_error value of target" do
        CoNeNe.srand(900)
        nn.init_weights
        layers = nn.layers
        result = nn.train_once( NArray.cast( [1.0, 0.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) )
        result.should be_within(1e-6).of 0.367103
      end

      it "modifies outputs" do
        CoNeNe.srand(900)
        nn.init_weights
        layers = nn.layers
        nn.train_once( NArray.cast( [1.0, 0.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) )
        layers[0].output.should be_narray_like NArray[ 0.778442, -0.774772, 0.0877405, 0.712681 ]
        layers[1].output.should be_narray_like NArray[ 0.39411 ]
      end

      it "modifies weights" do
        CoNeNe.srand(900)

        nn.init_weights
        layers = nn.layers
        w1 = layers[0].weights.clone
        w2 = layers[1].weights.clone

        nn.train_once( NArray.cast( [1.0, 0.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) )

        layers[0].weights.should_not be_narray_like w1
        layers[1].weights.should_not be_narray_like w2
      end

      it "can learn xor when run repeatedly" do
        ms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn.run xin
          ms_total += nn.ms_error( xtarg )
        end
        ms_total /= 4

        tries = 0
        while ( tries < 10 && ! xor_test(nn) )
          tries += 1
          nn.init_weights
          2000.times do
            xor_train_set.each do | xin, xtarg |
              nn.train_once xin, xtarg
            end
          end
        end

        after_ms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn.run xin
          after_ms_total += nn.ms_error( xtarg )
        end
        after_ms_total /= 4

        after_ms_total.should be < ms_total

        nn.run( NArray.cast( [-1.0, -1.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
        nn.run( NArray.cast( [-1.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn.run( NArray.cast( [1.0, -1.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn.run( NArray.cast( [1.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
      end

      it "can learn xor with 2 hidden layers" do
        ms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn2.run xin
          ms_total += nn2.ms_error( xtarg )
        end
        ms_total /= 4

        tries = 0
        while ( tries < 10 && ! xor_test(nn2) )
          tries += 1
          nn2.init_weights
          4000.times do
            xor_train_set.each do | xin, xtarg |
              nn2.train_once xin, xtarg
            end
          end
        end

        after_ms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn2.run xin
          after_ms_total += nn2.ms_error( xtarg )
        end
        after_ms_total /= 4

        after_ms_total.should be < ms_total

        nn2.run( NArray.cast( [-1.0, -1.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
        nn2.run( NArray.cast( [-1.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn2.run( NArray.cast( [1.0, -1.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn2.run( NArray.cast( [1.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
      end
    end
  end

  describe "first layer" do
    let( :nn ) { CoNeNe::MLP::Network.new( 2, [4], 1 ) }
    let( :first_layer ) { nn.layers.first }
    let( :second_layer ) { nn.layers.last }

    it "does not accept a new input_layer" do
      alt_layer = CoNeNe::MLP::Layer.new( 3, 2 )
      expect { first_layer.attach_input_layer( alt_layer ) }.to raise_error
    end

    it "can be cloned, and the clone does accept a new input_layer" do
      clone_of_first_layer = first_layer.clone
      alt_layer = CoNeNe::MLP::Layer.new( 3, 2 )
      expect { clone_of_first_layer.attach_input_layer( alt_layer ) }.to_not raise_error
      expect { first_layer.attach_input_layer( alt_layer ) }.to raise_error
    end
  end

end
