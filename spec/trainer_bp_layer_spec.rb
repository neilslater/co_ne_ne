require 'helpers'

describe RuNeNe::Trainer::BPLayer do
  describe "class methods" do
    describe "#new" do
      it "creates a new layer" do
        expect( RuNeNe::Trainer::BPLayer.new( 5, 5 ) ).to be_a RuNeNe::Trainer::BPLayer
      end
    end
  end
end
