require 'helpers'

describe CoNeNe::MLP::NLayer do
  describe "class methods" do
    describe "#new" do
      it "creates a new layer" do
        CoNeNe::MLP::NLayer.new( 2, 1 ).should be_a CoNeNe::MLP::NLayer
      end
    end
  end
end
