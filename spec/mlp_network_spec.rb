require 'helpers'

describe CoNeNe::MLP::ZNetwork do
  describe "class methods" do
    describe "#new" do
      it "creates a new network" do
        CoNeNe::MLP::ZNetwork.new( 2, [4], 1 ).should be_a CoNeNe::MLP::ZNetwork
      end
    end
  end
end
