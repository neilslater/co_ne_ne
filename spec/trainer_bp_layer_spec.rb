require 'helpers'

describe RuNeNe::Trainer::BPLayer do
  describe "class methods" do
    describe "#new" do
      it "creates a new backprop trainer for a layer" do
        expect( RuNeNe::Trainer::BPLayer.new( :num_inputs => 5, :num_outputs => 5 ) ).to be_a RuNeNe::Trainer::BPLayer
      end

      it "refuses to create new trainers for bad parameters" do
        expect { RuNeNe::Trainer::BPLayer.new( 0 ) }.to raise_error TypeError
        expect { RuNeNe::Trainer::BPLayer.new( :num_inputs => 3 ) }.to raise_error ArgumentError
        expect { RuNeNe::Trainer::BPLayer.new( :num_outputs => 3 ) }.to raise_error ArgumentError
        expect { RuNeNe::Trainer::BPLayer.new( :num_inputs => "hello", :num_outputs => 3 ) }.to raise_error TypeError
      end

      it "creates expected sizes and defaults for arrays" do
        bpl = RuNeNe::Trainer::BPLayer.new( :num_inputs => 2, :num_outputs => 1 )
        expect( bpl.de_dz ).to be_narray_like NArray[ 0.0 ]
        expect( bpl.de_da ).to be_narray_like NArray[ 0.0, 0.0 ]
        expect( bpl.de_dw ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
        expect( bpl.de_dw_momentum ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
        expect( bpl.de_dw_rmsprop ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
      end

      it "uses conservative defaults for all learning params" do
        bpl = RuNeNe::Trainer::BPLayer.new( :num_inputs => 2, :num_outputs => 1 )
        expect( bpl.learning_rate ).to be_within( 1e-6 ).of 0.01
        expect( bpl.smoothing_rate ).to be_within( 1e-6 ).of 0.9
        expect( bpl.smoothing_type ).to be :none
        expect( bpl.weight_decay ).to eql 0.0
        expect( bpl.max_norm ).to eql 0.0
      end

      it "uses options hash to set learning params" do
        bpl = RuNeNe::Trainer::BPLayer.new( :num_inputs => 2, :num_outputs => 1,
            :learning_rate => 0.005, :smoothing_rate => 0.99, :weight_decay => 1e-4,
            :max_norm => 2.4, :smoothing_type => :rmsprop )
        expect( bpl.learning_rate ).to be_within( 1e-6 ).of 0.005
        expect( bpl.smoothing_rate ).to be_within( 1e-6 ).of 0.99
        expect( bpl.smoothing_type ).to be :rmsprop
        expect( bpl.weight_decay ).to be_within( 1e-8 ).of 1e-4
        expect( bpl.max_norm ).to be_within( 1e-6 ).of 2.4
      end
    end
  end
end
