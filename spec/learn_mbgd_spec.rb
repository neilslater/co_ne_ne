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

      it "accepts a hash description in place of a layer" do
        expect( RuNeNe::Learn::MBGD.new( [in_layer_learn, { :num_outputs => 1 } ] ) ).to be_a RuNeNe::Learn::MBGD
      end

      it "will not allow layer size mis-match" do
        expect {
          RuNeNe::Learn::MBGD.new( [in_layer_learn, { :num_inputs => 3, :num_outputs => 1 } ] )
        }.to raise_error RuntimeError
      end

      it "correctly determines layer num_inputs from previous layer num_outputs when using hash syntax" do
        learn = RuNeNe::Learn::MBGD.new( [ { :num_inputs => 2, :num_outputs => 4 },
          { :num_outputs => 7 }, { :num_outputs => 2 }  ]
        )
        expect( learn.layer(0).num_inputs ).to be 2
        expect( learn.layer(1).num_inputs ).to be 4
        expect( learn.layer(2).num_inputs ).to be 7

        expect( learn.layer(2).num_outputs ).to be 2
      end

      it "allows specification of gradient descent types using hash syntax" do
        learn = RuNeNe::Learn::MBGD.new( [
          { :num_inputs => 2, :num_outputs => 4, :gradient_descent_type => :nag },
          { :num_outputs => 1, :gradient_descent_type => :rmsprop } ]
        )
        expect( learn.layer(0).gradient_descent ).to be_a RuNeNe::GradientDescent::NAG
        expect( learn.layer(1).gradient_descent ).to be_a RuNeNe::GradientDescent::RMSProp
      end
    end

    describe "#from_nn_model" do
      before :each do
        @nn = RuNeNe::NNModel.new( [in_layer_nn, out_layer_nn] )
      end

      it "creates a new backprop trainer for a network" do
        expect( RuNeNe::Learn::MBGD.from_nn_model( @nn ) ).to be_a RuNeNe::Learn::MBGD
      end

      it "refuses to create new trainers for bad parameters" do
        expect { RuNeNe::Learn::MBGD.from_nn_model( @nn, 0 ) }.to raise_error TypeError

        # :de_dz not supported
        expect { RuNeNe::Learn::MBGD.from_nn_model( @nn,
            :de_dz => NArray[ 0.0, 0.0 ] ) }.to raise_error ArgumentError

        # :de_dw not supported
        expect { RuNeNe::Learn::MBGD.from_nn_model( @nn,
            :de_dw => NArray[ 0.0, 0.0, 0.1, 0.2 ] ) }.to raise_error ArgumentError

        # :gradient_descent not supported
        expect { RuNeNe::Learn::MBGD.from_nn_model( @nn,
            :gradient_descent => RuNeNe::GradientDescent::RMSProp.new( NArray[ [-0.1, 0.01, 0.001] ], 0.8, 3e-7 ) )
        }.to raise_error ArgumentError
      end

      it "creates expected sizes and defaults for arrays" do
        learn = RuNeNe::Learn::MBGD.from_nn_model( @nn )
        expect( learn.layer(0).de_dz ).to be_narray_like NArray[ 0.0, 0.0 ]
        expect( learn.layer(1).de_dz ).to be_narray_like NArray[ 0.0 ]

        expect( learn.layer(0).de_da ).to be_narray_like NArray[ 0.0, 0.0 ]
        expect( learn.layer(1).de_da ).to be_narray_like NArray[ 0.0, 0.0 ]

        expect( learn.layer(0).de_dw ).to be_narray_like NArray[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ]
        expect( learn.layer(1).de_dw ).to be_narray_like NArray[ [0.0, 0.0, 0.0] ]
      end

      it "uses options hash to set learning params" do
        learn = RuNeNe::Learn::MBGD.from_nn_model( @nn,
            :learning_rate => 0.005, :decay => 0.99, :weight_decay => 1e-4,
            :max_norm => 2.4, :gradient_descent_type => :rmsprop )

        learn.mbgd_layers.each do |bpl|
          expect( bpl.learning_rate ).to be_within( 1e-6 ).of 0.005
          expect( bpl.gradient_descent_type ).to be :rmsprop
          expect( bpl.weight_decay ).to be_within( 1e-8 ).of 1e-4
          expect( bpl.max_norm ).to be_within( 1e-6 ).of 2.4
          expect( bpl.gradient_descent ).to be_a RuNeNe::GradientDescent::RMSProp
          expect( bpl.gradient_descent.decay ).to be_within(1e-6).of 0.99
          expect( bpl.gradient_descent.epsilon ).to be_within(1e-12).of 1e-6
        end

        expect( learn.layer(0).gradient_descent.num_params ).to be 6
        expect( learn.layer(0).gradient_descent.av_squared_grads ).to be_narray_like NArray[ [ 1.0, 1.0, 1.0 ], [ 1.0, 1.0, 1.0 ] ]

        expect( learn.layer(1).gradient_descent.num_params ).to be 3
        expect( learn.layer(1).gradient_descent.av_squared_grads ).to be_narray_like NArray[ [ 1.0, 1.0, 1.0 ] ]
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

    describe "#set_meta_params" do
      before :each do
        @nn = RuNeNe::NNModel.new( [in_layer_nn, out_layer_nn] )
      end

      [:sgd, :nag, :rmsprop].each do |accel_type|
        context "with gradient_descent_type '#{accel_type}'" do
          before :each do
            @learn_subject = RuNeNe::Learn::MBGD.from_nn_model( @nn,
                  :learning_rate => 0.02, :weight_decay => 1e-3,
                  :max_norm => 1.5, :gradient_descent_type => accel_type )
          end

          it "can set learning_rate in all layers" do
            @learn_subject.set_meta_params( :learning_rate => 0.07 )
            expect( @learn_subject.layer(0).learning_rate ).to be_within( 1e-6 ).of 0.07
            expect( @learn_subject.layer(1).learning_rate ).to be_within( 1e-6 ).of 0.07
          end

          it "doesn't reset other values when setting learning_rate" do
            @learn_subject.set_meta_params( :learning_rate => 0.07 )
            expect( @learn_subject.layer(0).weight_decay ).to be_within( 1e-6 ).of 1e-3
            expect( @learn_subject.layer(1).weight_decay ).to be_within( 1e-6 ).of 1e-3
            expect( @learn_subject.layer(0).max_norm ).to be_within( 1e-6 ).of 1.5
            expect( @learn_subject.layer(1).max_norm ).to be_within( 1e-6 ).of 1.5
          end

          it "can set weight_decay in all layers" do
            @learn_subject.set_meta_params( :weight_decay => 0.003 )
            expect( @learn_subject.layer(0).weight_decay ).to be_within( 1e-6 ).of 0.003
            expect( @learn_subject.layer(1).weight_decay ).to be_within( 1e-6 ).of 0.003
          end

          it "can set max_norm in all layers" do
            @learn_subject.set_meta_params( :max_norm => 0.75 )
            expect( @learn_subject.layer(0).max_norm ).to be_within( 1e-6 ).of 0.75
            expect( @learn_subject.layer(1).max_norm ).to be_within( 1e-6 ).of 0.75
          end

          case accel_type
          when :nag
            it "can set momentum in all layers" do
              @learn_subject.set_meta_params( :momentum => 0.55 )
              expect( @learn_subject.layer(0).gradient_descent.momentum ).to  be_within( 1e-6 ).of 0.55
              expect( @learn_subject.layer(1).gradient_descent.momentum ).to  be_within( 1e-6 ).of 0.55
            end
            it "doesn't reset momentum when setting learning_rate" do
              @learn_subject.set_meta_params( :learning_rate => 0.07 )
              expect( @learn_subject.layer(0).gradient_descent.momentum ).to  be_within( 1e-6 ).of 0.9
              expect( @learn_subject.layer(1).gradient_descent.momentum ).to  be_within( 1e-6 ).of 0.9
            end

          when :rmsprop
            it "can set decay in all layers" do
              @learn_subject.set_meta_params( :decay => 0.99 )
              expect( @learn_subject.layer(0).gradient_descent.decay ).to  be_within( 1e-6 ).of 0.99
              expect( @learn_subject.layer(1).gradient_descent.decay ).to  be_within( 1e-6 ).of 0.99
            end
            it "can set epsilon" do
              @learn_subject.set_meta_params( :epsilon => 0.001 )
              expect( @learn_subject.layer(0).gradient_descent.epsilon ).to  be_within( 1e-9 ).of 0.001
              expect( @learn_subject.layer(1).gradient_descent.epsilon ).to  be_within( 1e-9 ).of 0.001
            end
            it "doesn't reset decay or epsilon when setting learning_rate" do
              @learn_subject.set_meta_params( :learning_rate => 0.07 )
              expect( @learn_subject.layer(0).gradient_descent.decay ).to  be_within( 1e-6 ).of 0.9
              expect( @learn_subject.layer(0).gradient_descent.epsilon ).to  be_within( 1e-9 ).of 1e-6
              expect( @learn_subject.layer(1).gradient_descent.decay ).to  be_within( 1e-6 ).of 0.9
              expect( @learn_subject.layer(1).gradient_descent.epsilon ).to  be_within( 1e-9 ).of 1e-6
            end
          end
        end
      end
    end

    describe "#train_one_batch" do
      before :each do
        RuNeNe.srand( 1_000_000 )
        @nn = RuNeNe::NNModel.new( [in_layer_nn, out_layer_nn] )
        @xor_inputs = NArray.cast( [ [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0] ], 'sfloat' )
        @xor_targets = NArray.cast( [ [0.0], [1.0], [1.0], [0.0] ], 'sfloat' )
        @data = RuNeNe::DataSet.new( @xor_inputs, @xor_targets )
      end

      [:sgd, :nag, :rmsprop].each do |accel_type|
        context "with gradient_descent_type '#{accel_type}'" do
          before :each do
            @learn_subject = RuNeNe::Learn::MBGD.from_nn_model( @nn,
                  :learning_rate => 0.02, :weight_decay => 1e-3,
                  :max_norm => 1.5, :gradient_descent_type => accel_type )
          end

          it "returns a loss value" do
            loss = @learn_subject.train_one_batch( @nn, @data, :mse, 4 )
            expect( loss ).to be_within( 0.00001 ).of 0.111272
          end

        end
      end
    end
  end
end
