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

      it "plays nicely with Ruby's garbage collection" do
        layer = nil
        50000.times do
          layer = CoNeNe::MLP::NLayer.new( rand(100)+1, rand(50)+1 )
        end
        GC.start
        sleep 0.5
        5000.times do
          layer = CoNeNe::MLP::NLayer.new( rand(100)+1, rand(50)+1 )
        end
        sleep 0.5
      end
    end
  end
end
