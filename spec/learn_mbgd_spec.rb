require 'helpers'

describe RuNeNe::Learn::MBGD do
  let( :in_layer_nn ) { RuNeNe::Layer::FeedForward.new( 2, 2 ) }
  let( :out_layer_nn ) { RuNeNe::Layer::FeedForward.new( 2, 1 ) }
  let( :xor_nn ) { RuNeNe::NNModel.new( [in_layer_xor, out_layer_xor] ) }

  let( :in_layer_learn ) { RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 2, :num_outputs => 2 ) }
  let( :out_layer_learn ) { RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 2, :num_outputs => 1 ) }

  describe "class methods" do
    describe "#new" do
      it "creates a new Learn::MBGD" do
        expect( RuNeNe::Learn::MBGD.new( [in_layer_learn,out_layer_learn] ) ).to be_a RuNeNe::Learn::MBGD
      end

      it "refuses to create new Learn::MBGD for bad parameters" do
        expect { RuNeNe::Learn::MBGD.new( 17 ) }.to raise_error TypeError
        expect { RuNeNe::Learn::MBGD.new( [] ) }.to raise_error ArgumentError
        expect { RuNeNe::Learn::MBGD.new( [in_layer_learn,nil,out_layer_learn] ) }.to raise_error TypeError
      end
    end

    describe "with Marshal" do
      it "can save and retrieve a Learn::MBGD, preserving layer properties" do
        orig_mbgd = RuNeNe::Learn::MBGD.new( [in_layer_learn,out_layer_learn] )
        saved = Marshal.dump( orig_mbgd )
        copy_mbgd = Marshal.load( saved )

        expect( copy_mbgd ).to_not be orig_mbgd
        orig_layers = orig_mbgd.mbgd_layers
        copy_layers = copy_mbgd.mbgd_layers
        expect( copy_mbgd.num_inputs ).to eql orig_mbgd.num_inputs
        expect( copy_mbgd.num_outputs ).to eql orig_mbgd.num_outputs

        orig_layers.zip(copy_layers).each do |orig_layer, copy_layer|
          expect( copy_layer ).to_not be orig_layer
          expect( copy_layer.num_inputs ).to eql orig_layer.num_inputs
          expect( copy_layer.num_outputs ).to eql orig_layer.num_outputs
          expect( copy_layer.gradient_descent ).to be_a RuNeNe::GradientDescent::SGD
        end
      end
    end
  end

  describe "instance methods" do
    before :each do
      @mbgd = RuNeNe::Learn::MBGD.new( [in_layer_learn,out_layer_learn] )
    end

    describe "clone" do
      it "should make a simple copy of number of inputs and outputs" do
        copy = @mbgd.clone
        expect( copy.num_inputs ).to eql @mbgd.num_inputs
        expect( copy.num_outputs ).to eql @mbgd.num_outputs
      end

      it "should make a deep copy of layers" do
        copy = @mbgd.clone
        expect( copy.num_inputs ).to eql @mbgd.num_inputs
        expect( copy.num_outputs ).to eql @mbgd.num_outputs

        orig_layers = @mbgd.mbgd_layers
        copy_layers = copy.mbgd_layers

        orig_layers.zip(copy_layers).each do |orig_layer, copy_layer|
          expect( copy_layer ).to_not be orig_layer
          expect( copy_layer.num_inputs ).to eql orig_layer.num_inputs
          expect( copy_layer.num_outputs ).to eql orig_layer.num_outputs
          expect( copy_layer.gradient_descent ).to be_a RuNeNe::GradientDescent::SGD
        end
      end
    end

    describe "#layer" do
      it "accesses individual layer object from mbgd_layers" do
        expect( @mbgd.layer(0) ).to be  @mbgd.mbgd_layers[0]
        expect( @mbgd.layer(1) ).to be  @mbgd.mbgd_layers[1]
      end
    end
  end
end
