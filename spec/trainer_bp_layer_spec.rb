require 'helpers'

describe RuNeNe::Trainer::BPLayer do
  describe "class methods" do
    describe "#new" do
      it "creates a new backprop trainer for a layer" do
        expect( RuNeNe::Trainer::BPLayer.new( 5, 5 ) ).to be_a RuNeNe::Trainer::BPLayer
      end

      it "refuses to create new trainers for bad parameters" do
        expect { RuNeNe::Trainer::BPLayer.new( 0, 5 ) }.to raise_error ArgumentError
        expect { RuNeNe::Trainer::BPLayer.new( 3, -1 ) }.to raise_error ArgumentError
        expect { RuNeNe::Trainer::BPLayer.new( "hello", 2 ) }.to raise_error TypeError
      end
    end
  end
end
