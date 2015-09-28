require 'helpers'

describe RuNeNe::NNModel do
  let( :in_layer_xor ) { RuNeNe::Layer::FeedForward.new( 2, 2 ) }
  let( :out_layer_xor ) { RuNeNe::Layer::FeedForward.new( 2, 1 ) }

  describe "class methods" do
    describe "#new" do
      it "creates a new nn_model" do
        expect( RuNeNe::NNModel.new( [in_layer_xor, out_layer_xor] ) ).to be_a RuNeNe::NNModel
      end

      it "refuses to create new nn_models for bad parameters" do
        expect { RuNeNe::NNModel.new( 17 ) }.to raise_error TypeError
        expect { RuNeNe::NNModel.new( [] ) }.to raise_error ArgumentError
        expect { RuNeNe::NNModel.new( [in_layer_xor,nil,out_layer_xor] ) }.to raise_error TypeError
      end

      it "accepts a hash description in place of a layer" do
        expect( RuNeNe::NNModel.new( [in_layer_xor, { :num_outputs => 1 } ] ) ).to be_a RuNeNe::NNModel
      end

      it "will not allow layer size mis-match" do
        expect {
          RuNeNe::NNModel.new( [in_layer_xor, { :num_inputs => 3, :num_outputs => 1 } ] )
        }.to raise_error RuntimeError
      end

      it "correctly determines layer num_inputs from previous layer num_outputs when using hash syntax" do
        nn = RuNeNe::NNModel.new( [ { :num_inputs => 2, :num_outputs => 4},
          { :num_outputs => 7 }, { :num_outputs => 2 }  ]
        )
        expect( nn.layers[0].num_inputs ).to be 2
        expect( nn.layers[1].num_inputs ).to be 4
        expect( nn.layers[2].num_inputs ).to be 7

        expect( nn.layers[2].num_outputs ).to be 2
      end

      it "allows specification of transfer functions using hash syntax" do
        nn = RuNeNe::NNModel.new( [
          { :num_inputs => 2, :num_outputs => 4, :transfer => :softmax },
          { :num_outputs => 1, :transfer => :tanh } ]
        )
        expect( nn.layers[0].transfer ).to be RuNeNe::Transfer::Softmax
        expect( nn.layers[1].transfer ).to be RuNeNe::Transfer::TanH
      end
    end

    describe "with Marshal" do
      it "can save and retrieve a nn_model, preserving layer properties" do
        orig_nn_model = RuNeNe::NNModel.new( [in_layer_xor, out_layer_xor] )
        saved = Marshal.dump( orig_nn_model )
        copy_nn_model = Marshal.load( saved )

        expect( copy_nn_model ).to_not be orig_nn_model
        orig_layers = orig_nn_model.layers
        copy_layers = copy_nn_model.layers
        expect( copy_nn_model.num_inputs ).to eql orig_nn_model.num_inputs
        expect( copy_nn_model.num_outputs ).to eql orig_nn_model.num_outputs

        orig_layers.zip(copy_layers).each do |orig_layer, copy_layer|
          expect( copy_layer ).to_not be orig_layer
          expect( copy_layer.num_inputs ).to eql orig_layer.num_inputs
          expect( copy_layer.num_outputs ).to eql orig_layer.num_outputs
          expect( copy_layer.transfer ).to be RuNeNe::Transfer::Sigmoid
          expect( copy_layer.weights ).to be_narray_like orig_layer.weights
        end
      end
    end
  end

  describe "instance methods" do
    before :each do
      @nn = RuNeNe::NNModel.new( [in_layer_xor, out_layer_xor] )
    end

    describe "clone" do
      it "should make a simple copy of number of inputs and outputs" do
        copy = @nn.clone
        expect( copy.num_inputs ).to eql @nn.num_inputs
        expect( copy.num_outputs ).to eql @nn.num_outputs
      end

      it "should make a deep copy of layers" do
        copy = @nn.clone
        expect( copy.num_inputs ).to eql @nn.num_inputs
        expect( copy.num_outputs ).to eql @nn.num_outputs

        orig_layers = @nn.layers
        copy_layers = copy.layers

        orig_layers.zip(copy_layers).each do |orig_layer, copy_layer|
          expect( copy_layer ).to_not be orig_layer
          expect( copy_layer.num_inputs ).to eql orig_layer.num_inputs
          expect( copy_layer.num_outputs ).to eql orig_layer.num_outputs
          expect( copy_layer.transfer ).to be RuNeNe::Transfer::Sigmoid
          expect( copy_layer.weights ).to be_narray_like orig_layer.weights
        end
      end
    end

    describe "#layer" do
      it "accesses individual layer object from layers" do
        expect( @nn.layer(0) ).to be  @nn.layers[0]
        expect( @nn.layer(1) ).to be  @nn.layers[1]
      end
    end

    describe "#init_weights" do
      before :each do
        RuNeNe.srand(800)
      end

      it "should set weights to normal distribution by default" do
        @nn.init_weights
        expect( @nn.layers[0].weights ).to be_narray_like NArray[
          [ 0.290879, -0.143323, -0.206298 ],
          [ 0.145002, -0.409508, 0.210244 ] ]
        expect( @nn.layers[1].weights ).to be_narray_like NArray[
          [ -0.0573267, -0.134014, 0.048779 ] ]
      end

      it "should accept an optional multiplier" do
        @nn.init_weights( 0.1 )
        expect( @nn.layers[0].weights ).to be_narray_like NArray[
          [ 0.0290879, -0.0143323, -0.0206298 ],
          [ 0.0145002, -0.0409508, 0.0210244 ] ]
        expect( @nn.layers[1].weights ).to be_narray_like NArray[
          [ -0.00573267, -0.0134014, 0.0048779 ] ]
      end

      it "returns self" do
        expect( @nn.init_weights() ).to be @nn
        expect( @nn.init_weights( 2.5 ) ).to be @nn
      end
    end

    describe "#run" do
      before :each do
        RuNeNe.srand(800)
        @nn.init_weights
      end

      it "should produce an expected output" do
        result = @nn.run( NArray.cast( [-0.5, 0.7], 'sfloat' ) )
        expect( result ).to be_narray_like NArray[ 0.491116 ]

        result = @nn.run( NArray.cast( [0.5, -0.7], 'sfloat' ) )
        expect( result ).to be_narray_like NArray[ 0.483497 ]
      end

      it "sets activations in each layer" do
        @nn.run( NArray.cast( [-0.5, 0.7], 'sfloat' ) )
        expect( @nn.activations(0) ).to be_narray_like NArray[ 0.38887, 0.46284 ]
        expect( @nn.activations(1) ).to be_narray_like NArray[ 0.491116 ]

        @nn.run( NArray.cast( [0.5, -0.7], 'sfloat' ) )
        expect( @nn.activations(0) ).to be_narray_like NArray[ 0.509866, 0.638625 ]
        expect( @nn.activations(1) ).to be_narray_like NArray[ 0.483497 ]
      end

      it "should refuse to run for bad inputs" do
        expect { @nn.run( NArray.cast( [-0.5 ], 'sfloat' ) ) }.to raise_error ArgumentError
        expect { @nn.run( NArray.cast( [-0.5,-0.5,-0.5 ], 'sfloat' ) ) }.to raise_error ArgumentError
        expect { @nn.run( :hello ) }.to raise_error TypeError
      end
    end
  end
end
