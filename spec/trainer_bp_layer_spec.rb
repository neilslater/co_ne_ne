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
        expect { RuNeNe::Trainer::BPLayer.new( :num_outputs => 3, :num_inputs => 3,
            :de_dz => "Fish" ) }.to raise_error TypeError

        # :de_dz has wrong number of elements here
        expect { RuNeNe::Trainer::BPLayer.new( :num_outputs => 3, :num_inputs => 3,
            :de_dz => NArray[ 0.0, 0.0 ] ) }.to raise_error ArgumentError

        # :de_dw has wrong rank here
        expect { RuNeNe::Trainer::BPLayer.new( :num_outputs => 1, :num_inputs => 3,
            :de_dw => NArray[ 0.0, 0.0, 0.1, 0.2 ] ) }.to raise_error ArgumentError
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
        expect( bpl.gd_accel_rate ).to be_within( 1e-6 ).of 0.9
        expect( bpl.gd_accel_type ).to be :none
        expect( bpl.weight_decay ).to eql 0.0
        expect( bpl.max_norm ).to eql 0.0
      end

      it "uses options hash to set learning params" do
        bpl = RuNeNe::Trainer::BPLayer.new( :num_inputs => 2, :num_outputs => 1,
            :learning_rate => 0.005, :gd_accel_rate => 0.99, :weight_decay => 1e-4,
            :max_norm => 2.4, :gd_accel_type => :rmsprop )
        expect( bpl.learning_rate ).to be_within( 1e-6 ).of 0.005
        expect( bpl.gd_accel_rate ).to be_within( 1e-6 ).of 0.99
        expect( bpl.gd_accel_type ).to be :rmsprop
        expect( bpl.weight_decay ).to be_within( 1e-8 ).of 1e-4
        expect( bpl.max_norm ).to be_within( 1e-6 ).of 2.4
      end

      it "uses options hash to set narrays" do
        bpl = RuNeNe::Trainer::BPLayer.new( :num_inputs => 2, :num_outputs => 1,
            :de_dz => NArray[ 0.2 ], :de_da => NArray[ 0.1, 0.1 ], :de_dw => NArray[ [-0.1, 0.01, 0.001] ],
            :de_dw_momentum => NArray[ [0.1, -0.01, -0.001] ], :de_dw_rmsprop => NArray[ [-0.2, 0.02, 0.002] ]
            )
        expect( bpl.de_dz ).to be_narray_like NArray[ 0.2 ]
        expect( bpl.de_da ).to be_narray_like NArray[ 0.1, 0.1 ]
        expect( bpl.de_dw ).to be_narray_like NArray[ [-0.1, 0.01, 0.001] ]
        expect( bpl.de_dw_momentum ).to be_narray_like NArray[ [0.1, -0.01, -0.001] ]
        expect( bpl.de_dw_rmsprop ).to be_narray_like NArray[ [-0.2, 0.02, 0.002] ]
      end
    end

    describe "#from_layer" do
      before :each do
        @layer = RuNeNe::Layer::FeedForward.new( 2, 1 )
      end

      it "creates a new backprop trainer for a layer" do
        expect( RuNeNe::Trainer::BPLayer.from_layer( @layer ) ).to be_a RuNeNe::Trainer::BPLayer
      end

      it "refuses to create new trainers for bad parameters" do
        expect { RuNeNe::Trainer::BPLayer.from_layer( @layer, 0 ) }.to raise_error TypeError
        expect { RuNeNe::Trainer::BPLayer.from_layer( @layer,
            :de_dz => "Fish" ) }.to raise_error TypeError

        # :de_dz has wrong number of elements here
        expect { RuNeNe::Trainer::BPLayer.from_layer( @layer,
            :de_dz => NArray[ 0.0, 0.0 ] ) }.to raise_error ArgumentError

        # :de_dw has wrong rank here
        expect { RuNeNe::Trainer::BPLayer.from_layer( @layer,
            :de_dw => NArray[ 0.0, 0.0, 0.1, 0.2 ] ) }.to raise_error ArgumentError
      end

      it "creates expected sizes and defaults for arrays" do
        bpl = RuNeNe::Trainer::BPLayer.from_layer( @layer )
        expect( bpl.de_dz ).to be_narray_like NArray[ 0.0 ]
        expect( bpl.de_da ).to be_narray_like NArray[ 0.0, 0.0 ]
        expect( bpl.de_dw ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
        expect( bpl.de_dw_momentum ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
        expect( bpl.de_dw_rmsprop ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
      end

      it "uses options hash to set learning params" do
        bpl = RuNeNe::Trainer::BPLayer.from_layer( @layer,
            :learning_rate => 0.005, :gd_accel_rate => 0.99, :weight_decay => 1e-4,
            :max_norm => 2.4, :gd_accel_type => :rmsprop )
        expect( bpl.learning_rate ).to be_within( 1e-6 ).of 0.005
        expect( bpl.gd_accel_rate ).to be_within( 1e-6 ).of 0.99
        expect( bpl.gd_accel_type ).to be :rmsprop
        expect( bpl.weight_decay ).to be_within( 1e-8 ).of 1e-4
        expect( bpl.max_norm ).to be_within( 1e-6 ).of 2.4
      end

      it "uses options hash to set narrays" do
        bpl = RuNeNe::Trainer::BPLayer.from_layer( @layer,
            :de_dz => NArray[ 0.2 ], :de_da => NArray[ 0.1, 0.1 ], :de_dw => NArray[ [-0.1, 0.01, 0.001] ],
            :de_dw_momentum => NArray[ [0.1, -0.01, -0.001] ], :de_dw_rmsprop => NArray[ [-0.2, 0.02, 0.002] ]
            )
        expect( bpl.de_dz ).to be_narray_like NArray[ 0.2 ]
        expect( bpl.de_da ).to be_narray_like NArray[ 0.1, 0.1 ]
        expect( bpl.de_dw ).to be_narray_like NArray[ [-0.1, 0.01, 0.001] ]
        expect( bpl.de_dw_momentum ).to be_narray_like NArray[ [0.1, -0.01, -0.001] ]
        expect( bpl.de_dw_rmsprop ).to be_narray_like NArray[ [-0.2, 0.02, 0.002] ]
      end
    end

    describe "with Marshal" do
      it "can save and retrieve a training layer, preserving all property values" do
        orig_bpl = RuNeNe::Trainer::BPLayer.new( :num_inputs => 2, :num_outputs => 2,
            :learning_rate => 0.003, :gd_accel_rate => 0.97, :weight_decay => 2e-3,
            :max_norm => 1.1, :gd_accel_type => :momentum,
            :de_dz => NArray[ 0.25, 0.5 ], :de_da => NArray[ 0.13, 0.14 ],
            :de_dw => NArray[ [-0.1, 0.01, 0.001], [0.6, 0.5, 0.4] ],
            :de_dw_momentum => NArray[ [0.1, -0.01, -0.001], [0.55, 0.35, 0.14] ],
            :de_dw_rmsprop => NArray[ [-0.2, 0.02, 0.002], [0.63, 0.25, 0.24] ]
        )
        saved = Marshal.dump( orig_bpl )
        copy_bpl = Marshal.load( saved )

        expect( copy_bpl ).to_not be orig_bpl
        expect( copy_bpl.num_inputs ).to be 2
        expect( copy_bpl.num_outputs ).to be 2
        expect( copy_bpl.learning_rate ).to be_within( 1e-6 ).of 0.003
        expect( copy_bpl.gd_accel_rate ).to be_within( 1e-6 ).of 0.97
        expect( copy_bpl.gd_accel_type ).to be :momentum
        expect( copy_bpl.weight_decay ).to be_within( 1e-8 ).of 2e-3
        expect( copy_bpl.max_norm ).to be_within( 1e-6 ).of 1.1

        expect( copy_bpl.de_dz ).to be_narray_like orig_bpl.de_dz
        expect( copy_bpl.de_da ).to be_narray_like orig_bpl.de_da
        expect( copy_bpl.de_dw ).to be_narray_like orig_bpl.de_dw
        expect( copy_bpl.de_dw_momentum ).to be_narray_like orig_bpl.de_dw_momentum
        expect( copy_bpl.de_dw_rmsprop ).to be_narray_like orig_bpl.de_dw_rmsprop
      end
    end
  end

  describe "instance methods" do
    before :each do
      @bpl = RuNeNe::Trainer::BPLayer.new( :num_inputs => 3, :num_outputs => 2,
            :learning_rate => 0.007, :gd_accel_rate => 0.95, :weight_decay => 1e-3,
            :max_norm => 1.5, :gd_accel_type => :momentum )
    end

    describe "#clone" do
      it "makes copies of learning params" do
        @copy = @bpl.clone
        expect( @copy ).to_not be @bpl
        expect( @copy.num_inputs ).to be 3
        expect( @copy.num_outputs ).to be 2

        expect( @copy.learning_rate ).to be_within( 1e-6 ).of 0.007
        expect( @copy.gd_accel_rate ).to be_within( 1e-6 ).of 0.95
        expect( @copy.gd_accel_type ).to be :momentum
        expect( @copy.weight_decay ).to be_within( 1e-8 ).of 1e-3
        expect( @copy.max_norm ).to be_within( 1e-6 ).of 1.5
      end

      it "makes deep copy of layer training data" do
        @copy = @bpl.clone
        expect( @copy ).to_not be @bpl

        expect( @copy.de_dz ).to_not be @bpl.de_dz
        expect( @copy.de_da ).to_not be @bpl.de_da
        expect( @copy.de_dw ).to_not be @bpl.de_dw
        expect( @copy.de_dw_momentum ).to_not be @bpl.de_dw_momentum
        expect( @copy.de_dw_rmsprop ).to_not be @bpl.de_dw_rmsprop

        expect( @copy.de_dz ).to be_narray_like @bpl.de_dz
        expect( @copy.de_da ).to be_narray_like @bpl.de_da
        expect( @copy.de_dw ).to be_narray_like @bpl.de_dw
        expect( @copy.de_dw_momentum ).to be_narray_like @bpl.de_dw_momentum
        expect( @copy.de_dw_rmsprop ).to be_narray_like @bpl.de_dw_rmsprop
      end
    end

    describe "#start_batch" do
      it "resets de_dw" do
        @bpl.de_dw[0,0] = 1.0
        @bpl.de_dw[1,0] = 2.0
        @bpl.de_dw[2,0] = 3.0
        @bpl.de_dw[3,1] = 4.0
        expect( @bpl.de_dw ).to be_narray_like NArray[[1.0,2.0,3.0,0.0],[0.0,0.0,0.0,4.0]]

        @bpl.start_batch

        expect( @bpl.de_dw ).to be_narray_like NArray[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
      end
    end
  end
end
