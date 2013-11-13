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
        if RUBY_DESCRIPTION.include? "rubinius"
          number_of_layers = 500
        end

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
end
