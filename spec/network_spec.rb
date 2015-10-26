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
    end
  end
end
