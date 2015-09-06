require 'helpers'

describe RuNeNe::Learn::MBGD::Layer do
  describe "class methods" do
    describe "#new" do
      it "creates a new backprop trainer for a layer" do
        expect( RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 5, :num_outputs => 5 ) ).to be_a RuNeNe::Learn::MBGD::Layer
      end

      it "refuses to create new trainers for bad parameters" do
        expect { RuNeNe::Learn::MBGD::Layer.new( 0 ) }.to raise_error TypeError
        expect { RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 3 ) }.to raise_error ArgumentError
        expect { RuNeNe::Learn::MBGD::Layer.new( :num_outputs => 3 ) }.to raise_error ArgumentError
        expect { RuNeNe::Learn::MBGD::Layer.new( :num_inputs => "hello", :num_outputs => 3 ) }.to raise_error TypeError
        expect { RuNeNe::Learn::MBGD::Layer.new( :num_outputs => 3, :num_inputs => 3,
            :de_dz => "Fish" ) }.to raise_error TypeError

        # :de_dz has wrong number of elements here
        expect { RuNeNe::Learn::MBGD::Layer.new( :num_outputs => 3, :num_inputs => 3,
            :de_dz => NArray[ 0.0, 0.0 ] ) }.to raise_error ArgumentError

        # :de_dw has wrong rank here
        expect { RuNeNe::Learn::MBGD::Layer.new( :num_outputs => 1, :num_inputs => 3,
            :de_dw => NArray[ 0.0, 0.0, 0.1, 0.2 ] ) }.to raise_error ArgumentError
      end

      it "creates expected sizes and defaults for arrays" do
        bpl = RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 2, :num_outputs => 1 )
        expect( bpl.de_dz ).to be_narray_like NArray[ 0.0 ]
        expect( bpl.de_da ).to be_narray_like NArray[ 0.0, 0.0 ]
        expect( bpl.de_dw ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
        expect( bpl.de_dw_stats_a ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
        expect( bpl.de_dw_stats_b ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
      end

      it "uses conservative defaults for all learning params" do
        bpl = RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 2, :num_outputs => 1 )
        expect( bpl.learning_rate ).to be_within( 1e-6 ).of 0.01
        expect( bpl.gd_accel_rate ).to be_within( 1e-6 ).of 0.9
        expect( bpl.gd_accel_type ).to be :none
        expect( bpl.weight_decay ).to eql 0.0
        expect( bpl.max_norm ).to eql 0.0
      end

      it "uses options hash to set learning params" do
        bpl = RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 2, :num_outputs => 1,
            :learning_rate => 0.005, :gd_accel_rate => 0.99, :weight_decay => 1e-4,
            :max_norm => 2.4, :gd_accel_type => :rmsprop )
        expect( bpl.learning_rate ).to be_within( 1e-6 ).of 0.005
        expect( bpl.gd_accel_rate ).to be_within( 1e-6 ).of 0.99
        expect( bpl.gd_accel_type ).to be :rmsprop
        expect( bpl.weight_decay ).to be_within( 1e-8 ).of 1e-4
        expect( bpl.max_norm ).to be_within( 1e-6 ).of 2.4
      end

      it "uses options hash to set narrays" do
        bpl = RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 2, :num_outputs => 1,
            :de_dz => NArray[ 0.2 ], :de_da => NArray[ 0.1, 0.1 ], :de_dw => NArray[ [-0.1, 0.01, 0.001] ],
            :de_dw_stats_a => NArray[ [0.1, -0.01, -0.001] ], :de_dw_stats_b => NArray[ [-0.2, 0.02, 0.002] ]
            )
        expect( bpl.de_dz ).to be_narray_like NArray[ 0.2 ]
        expect( bpl.de_da ).to be_narray_like NArray[ 0.1, 0.1 ]
        expect( bpl.de_dw ).to be_narray_like NArray[ [-0.1, 0.01, 0.001] ]
        expect( bpl.de_dw_stats_a ).to be_narray_like NArray[ [0.1, -0.01, -0.001] ]
        expect( bpl.de_dw_stats_b ).to be_narray_like NArray[ [-0.2, 0.02, 0.002] ]
      end
    end

    describe "#from_layer" do
      before :each do
        @layer = RuNeNe::Layer::FeedForward.new( 2, 1 )
      end

      it "creates a new backprop trainer for a layer" do
        expect( RuNeNe::Learn::MBGD::Layer.from_layer( @layer ) ).to be_a RuNeNe::Learn::MBGD::Layer
      end

      it "refuses to create new trainers for bad parameters" do
        expect { RuNeNe::Learn::MBGD::Layer.from_layer( @layer, 0 ) }.to raise_error TypeError
        expect { RuNeNe::Learn::MBGD::Layer.from_layer( @layer,
            :de_dz => "Fish" ) }.to raise_error TypeError

        # :de_dz has wrong number of elements here
        expect { RuNeNe::Learn::MBGD::Layer.from_layer( @layer,
            :de_dz => NArray[ 0.0, 0.0 ] ) }.to raise_error ArgumentError

        # :de_dw has wrong rank here
        expect { RuNeNe::Learn::MBGD::Layer.from_layer( @layer,
            :de_dw => NArray[ 0.0, 0.0, 0.1, 0.2 ] ) }.to raise_error ArgumentError
      end

      it "creates expected sizes and defaults for arrays" do
        bpl = RuNeNe::Learn::MBGD::Layer.from_layer( @layer )
        expect( bpl.de_dz ).to be_narray_like NArray[ 0.0 ]
        expect( bpl.de_da ).to be_narray_like NArray[ 0.0, 0.0 ]
        expect( bpl.de_dw ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
        expect( bpl.de_dw_stats_a ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
        expect( bpl.de_dw_stats_b ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
      end

      it "uses options hash to set learning params" do
        bpl = RuNeNe::Learn::MBGD::Layer.from_layer( @layer,
            :learning_rate => 0.005, :gd_accel_rate => 0.99, :weight_decay => 1e-4,
            :max_norm => 2.4, :gd_accel_type => :rmsprop )
        expect( bpl.learning_rate ).to be_within( 1e-6 ).of 0.005
        expect( bpl.gd_accel_rate ).to be_within( 1e-6 ).of 0.99
        expect( bpl.gd_accel_type ).to be :rmsprop
        expect( bpl.weight_decay ).to be_within( 1e-8 ).of 1e-4
        expect( bpl.max_norm ).to be_within( 1e-6 ).of 2.4
      end

      it "uses options hash to set narrays" do
        bpl = RuNeNe::Learn::MBGD::Layer.from_layer( @layer,
            :de_dz => NArray[ 0.2 ], :de_da => NArray[ 0.1, 0.1 ], :de_dw => NArray[ [-0.1, 0.01, 0.001] ],
            :de_dw_stats_a => NArray[ [0.1, -0.01, -0.001] ], :de_dw_stats_b => NArray[ [-0.2, 0.02, 0.002] ]
            )
        expect( bpl.de_dz ).to be_narray_like NArray[ 0.2 ]
        expect( bpl.de_da ).to be_narray_like NArray[ 0.1, 0.1 ]
        expect( bpl.de_dw ).to be_narray_like NArray[ [-0.1, 0.01, 0.001] ]
        expect( bpl.de_dw_stats_a ).to be_narray_like NArray[ [0.1, -0.01, -0.001] ]
        expect( bpl.de_dw_stats_b ).to be_narray_like NArray[ [-0.2, 0.02, 0.002] ]
      end
    end

    describe "with Marshal" do
      it "can save and retrieve a training layer, preserving all property values" do
        orig_bpl = RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 2, :num_outputs => 2,
            :learning_rate => 0.003, :gd_accel_rate => 0.97, :weight_decay => 2e-3,
            :max_norm => 1.1, :gd_accel_type => :momentum,
            :de_dz => NArray[ 0.25, 0.5 ], :de_da => NArray[ 0.13, 0.14 ],
            :de_dw => NArray[ [-0.1, 0.01, 0.001], [0.6, 0.5, 0.4] ],
            :de_dw_stats_a => NArray[ [0.1, -0.01, -0.001], [0.55, 0.35, 0.14] ],
            :de_dw_stats_b => NArray[ [-0.2, 0.02, 0.002], [0.63, 0.25, 0.24] ]
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
        expect( copy_bpl.de_dw_stats_a ).to be_narray_like orig_bpl.de_dw_stats_a
        expect( copy_bpl.de_dw_stats_b ).to be_narray_like orig_bpl.de_dw_stats_b
      end
    end
  end

  describe "instance methods" do
    before :each do
      RuNeNe.srand( 8243 )
      NArray.srand( 9063 )

      @bpl_momentum = RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 3, :num_outputs => 2,
            :learning_rate => 0.02, :gd_accel_rate => 0.95, :weight_decay => 1e-3,
            :max_norm => 1.5, :gd_accel_type => :momentum )
    end

    describe "#clone" do
      it "makes copies of learning params" do
        @copy = @bpl_momentum.clone
        expect( @copy ).to_not be @bpl_momentum
        expect( @copy.num_inputs ).to be 3
        expect( @copy.num_outputs ).to be 2

        expect( @copy.learning_rate ).to be_within( 1e-6 ).of 0.02
        expect( @copy.gd_accel_rate ).to be_within( 1e-6 ).of 0.95
        expect( @copy.gd_accel_type ).to be :momentum
        expect( @copy.momentum ).to be_within( 1e-6 ).of 0.95
        expect( @copy.weight_decay ).to be_within( 1e-8 ).of 1e-3
        expect( @copy.max_norm ).to be_within( 1e-6 ).of 1.5
      end

      it "makes deep copy of layer training data" do
        @copy = @bpl_momentum.clone
        expect( @copy ).to_not be @bpl_momentum

        expect( @copy.de_dz ).to_not be @bpl_momentum.de_dz
        expect( @copy.de_da ).to_not be @bpl_momentum.de_da
        expect( @copy.de_dw ).to_not be @bpl_momentum.de_dw
        expect( @copy.de_dw_stats_a ).to_not be @bpl_momentum.de_dw_stats_a
        expect( @copy.de_dw_stats_b ).to_not be @bpl_momentum.de_dw_stats_b

        expect( @copy.de_dz ).to be_narray_like @bpl_momentum.de_dz
        expect( @copy.de_da ).to be_narray_like @bpl_momentum.de_da
        expect( @copy.de_dw ).to be_narray_like @bpl_momentum.de_dw
        expect( @copy.de_dw_stats_a ).to be_narray_like @bpl_momentum.de_dw_stats_a
        expect( @copy.de_dw_stats_b ).to be_narray_like @bpl_momentum.de_dw_stats_b
      end
    end

    describe "#start_batch" do
      before :each do
        @ff_layer = RuNeNe::Layer::FeedForward.new( 3, 2 )
      end

      [:none, :momentum, :rmsprop].each do |accel_type|
        context "with gd_accel_type '#{accel_type}'" do
          before :each do
            @bpl_subject = RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 3, :num_outputs => 2,
                  :learning_rate => 0.02, :gd_accel_rate => 0.95, :weight_decay => 1e-3,
                  :max_norm => 1.5, :gd_accel_type => accel_type )
          end

          it "resets de_dw" do
            @bpl_subject.de_dw[0,0] = 1.0
            @bpl_subject.de_dw[1,0] = 2.0
            @bpl_subject.de_dw[2,0] = 3.0
            @bpl_subject.de_dw[3,1] = 4.0
            expect( @bpl_subject.de_dw ).to be_narray_like NArray[[1.0,2.0,3.0,0.0],[0.0,0.0,0.0,4.0]]

            @bpl_subject.start_batch( @ff_layer )

            expect( @bpl_subject.de_dw ).to be_narray_like NArray[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
          end

          case accel_type
          when :momentum
            it "defaults de_dw_stats_a to all zeroes for first batch" do
              @bpl_subject.start_batch( @ff_layer )
              expect( @bpl_subject.de_dw_stats_a ).to be_narray_like NArray[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
            end

            it "represents de_dw_stats_a as weight_update_velocity" do
              expect( @bpl_subject.weight_update_velocity ).to be @bpl_subject.de_dw_stats_a
            end
          end
        end
      end
    end

    describe "#backprop_for_output_layer" do
      before :each do
        @ff_layer = RuNeNe::Layer::FeedForward.new( 3, 2 )
        @bpl_momentum.start_batch( @ff_layer )
        @inputs = NArray[0.3,-0.5,0.9]
        @outputs = @ff_layer.run( @inputs )
        @targets = NArray[1.0,0.0]
      end

      it "updates de_dw" do
        expect( @bpl_momentum.de_dw ).to be_narray_like NArray[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
        @bpl_momentum.backprop_for_output_layer( @ff_layer, @inputs, @outputs, @targets, :logloss )
        expect( @bpl_momentum.de_dw ).to_not be_narray_like NArray[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
        expect( @bpl_momentum.de_dw ).to be_narray_like NArray[
          [ -0.0720577, 0.120096, -0.216173, -0.240192 ],
          [ 0.183035, -0.305059, 0.549105, 0.610117 ]
        ]
      end

      it "doesn't change layer weights" do
        before_weights = @ff_layer.weights.clone
        @bpl_momentum.backprop_for_output_layer( @ff_layer, @inputs, @outputs, @targets, :logloss )
        expect( @ff_layer.weights ).to be_narray_like( before_weights, 1e-16 )
      end
    end

    describe "#finish_batch" do
      [:none, :momentum, :rmsprop].each do |accel_type|
        context "with gd_accel_type '#{accel_type}'" do
          before :each do
            @bpl_subject = RuNeNe::Learn::MBGD::Layer.new( :num_inputs => 3, :num_outputs => 2,
                  :learning_rate => 0.02, :gd_accel_rate => 0.95, :weight_decay => 1e-3,
                  :max_norm => 1.5, :gd_accel_type => accel_type )
          end

          before :each do
            @ff_layer = RuNeNe::Layer::FeedForward.new( 3, 2 )
            @bpl_subject.start_batch( @ff_layer )
            @inputs = NArray[0.3,-0.5,0.9]
            @outputs = @ff_layer.run( @inputs )
            @targets = NArray[1.0,0.0]
            @bpl_subject.backprop_for_output_layer( @ff_layer, @inputs, @outputs, @targets, :logloss )
          end

          it "doesn't change de_dw" do
            @bpl_subject.finish_batch( @ff_layer )
            expect( @bpl_subject.de_dw ).to be_narray_like NArray[
              [ -0.0720577, 0.120096, -0.216173, -0.240192 ],
              [ 0.183035, -0.305059, 0.549105, 0.610117 ]
            ]
          end

          it "changes layer weights" do
            before_weights = @ff_layer.weights.clone
            @bpl_subject.finish_batch( @ff_layer )
            expect( @ff_layer.weights ).to_not be_narray_like before_weights
            expect( @ff_layer.weights ).to be_narray_like NArray[
              [ 0.306208, -0.549897, 0.520301, 0.326871 ],
              [ 0.823996, -0.405873, -0.0354835, 0.00336937 ]
            ]
          end

          case accel_type
            when :momentum
            it "changes weight_update_velocity" do
              @bpl_subject.finish_batch( @ff_layer )
              expect( @bpl_subject.de_dw_stats_a ).to_not be_narray_like NArray[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]
              expect( @bpl_subject.weight_update_velocity ).to be_narray_like NArray[
                [ 0.00144115, -0.00240192, 0.00432346, 0.00480385 ],
                [ -0.0036607, 0.00610117, -0.0109821, -0.0122023 ]
              ]
              expect( @bpl_subject.weight_update_velocity ).to be @bpl_subject.de_dw_stats_a
            end
          end
        end
      end
    end
  end
end