require 'helpers'

describe RuNeNe::Network do
  let( :in_layer_xor ) { RuNeNe::Layer::FeedForward.new( 2, 2 ) }
  let( :out_layer_xor ) { RuNeNe::Layer::FeedForward.new( 2, 1 ) }
  let( :nn_model ) { RuNeNe::NNModel.new( [in_layer_xor, out_layer_xor] ) }
  let( :learn_mbgd ) { RuNeNe::Learn::MBGD.from_nn_model( nn_model ) }

  describe "class methods" do
    describe "#new" do
      it "creates a new network" do
        expect( RuNeNe::Network.new( nn_model, learn_mbgd ) ).to be_a RuNeNe::Network
      end

      it "rejects bad params" do
        expect { RuNeNe::Network.new( learn_mbgd, nn_model ) }.to raise_error TypeError
        expect { RuNeNe::Network.new( nn_model, nil ) }.to raise_error TypeError
        expect { RuNeNe::Network.new( nil, learn_mbgd ) }.to raise_error TypeError
        expect { RuNeNe::Network.new( in_layer_xor, learn_mbgd ) }.to raise_error TypeError
      end

      it "uses the input params as new attributes" do
        nn = nn_model
        mbgd = learn_mbgd
        network = RuNeNe::Network.new( nn, mbgd )
        expect( network.nn_model ).to be nn
        expect( network.learn ).to be mbgd
      end
    end

    describe "with Marshal" do
      before :each do
        @network = RuNeNe::Network.new( nn_model, learn_mbgd )
        @saved_network = Marshal.dump( @network )
        @network_copy = Marshal.load( @saved_network )
      end

      it "saves and loads all weights data" do
        [0,1].each do |layer_id|
          expect( @network_copy.nn_model.layer(layer_id).weights ).to be_narray_like @network.nn_model.layer(layer_id).weights
          expect( @network_copy.learn.layer(layer_id).de_da ).to be_narray_like @network.learn.layer(layer_id).de_da
          expect( @network_copy.learn.layer(layer_id).de_dw ).to be_narray_like @network.learn.layer(layer_id).de_dw
        end
      end
    end
  end

  describe "instance methods" do
    before :each do
      @network = RuNeNe::Network.new( nn_model, learn_mbgd )
    end

    describe "#clone" do
      before :each do
        @network_copy = @network.clone
      end

      it "makes a deep copy" do
        expect( @network_copy.learn ).to_not be @network.learn
        expect( @network_copy.nn_model ).to_not be @network.nn_model

        [0,1].each do |layer_id|
          expect( @network_copy.nn_model.layer(layer_id).weights ).to_not be @network.nn_model.layer(layer_id).weights
          expect( @network_copy.learn.layer(layer_id).de_da ).to_not be @network.learn.layer(layer_id).de_da
          expect( @network_copy.learn.layer(layer_id).de_dw ).to_not be @network.learn.layer(layer_id).de_dw

          expect( @network_copy.nn_model.layer(layer_id).weights ).to be_narray_like @network.nn_model.layer(layer_id).weights
          expect( @network_copy.learn.layer(layer_id).de_da ).to be_narray_like @network.learn.layer(layer_id).de_da
          expect( @network_copy.learn.layer(layer_id).de_dw ).to be_narray_like @network.learn.layer(layer_id).de_dw
        end
      end
    end
  end
end
