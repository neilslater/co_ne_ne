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
  end

  describe "instance methods" do
    let(:layer) { CoNeNe::MLP::NLayer.new( 3, 2 ) }

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

      it "should accept a two params to select from a range" do
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
    end

  end
end
