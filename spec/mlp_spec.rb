require 'helpers'

describe CoNeNe::MLP::Network do
  describe "class methods" do
    describe "#new" do
      it "creates a new network" do
        CoNeNe::MLP::Network.new( 2, [4], 1 ).should be_a CoNeNe::MLP::Network
      end

      it "creates layers" do
        nn = CoNeNe::MLP::Network.new( 2, [4], 1 )
        nn.layers.should be_a Array
        nn.layers.count.should == 2

        nn.layers.first.should be_a CoNeNe::MLP::Layer
        nn.layers.first.num_inputs.should == 2
        nn.layers.first.num_outputs.should == 4

        nn.layers.last.should be_a CoNeNe::MLP::Layer
        nn.layers.last.num_inputs.should == 4
        nn.layers.last.num_outputs.should == 1
      end
    end
  end

  describe "instance methods" do
    let( :nn ) { CoNeNe::MLP::Network.new( 2, [4], 1 ) }
    let( :nn2 ) { CoNeNe::MLP::Network.new( 2, [5,3], 1 ) }
    let( :nn3 ) { CoNeNe::MLP::Network.new( 2, [4,3,2], 1 ) }
    let( :xor_train_set ) {
      [
        [  NArray.cast( [-1.0, -1.0], 'sfloat' ), NArray.cast( [0.0], 'sfloat' ) ],
        [  NArray.cast( [-1.0, 1.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) ],
        [  NArray.cast( [1.0, -1.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) ],
        [  NArray.cast( [1.0, 1.0], 'sfloat' ), NArray.cast( [0.0], 'sfloat' ) ]
      ]
    }

    describe "#init_weights" do
      it "generates new values for weights" do
        old_weights0 = nn.layers[0].weights.clone
        old_weights1 = nn.layers[1].weights.clone

        nn.init_weights
        nn.layers[0].weights.should_not be_narray_like old_weights0
        nn.layers[1].weights.should_not be_narray_like old_weights1
      end
    end

    describe "#run" do
      it "modifies output" do
        nn.output.should eq NArray.sfloat(1)
        nn.run( NArray.cast( [1.0, 0.0], 'sfloat' ) )
        nn.output.should_not eq NArray.sfloat(1)
      end
    end

    describe "#train_once" do
      it "modifies output" do
        nn.train_once( NArray.cast( [1.0, 0.0], 'sfloat' ), NArray.cast( [1.0], 'sfloat' ) )
        nn.output.should_not eq NArray.sfloat(1)
      end

      it "can learn xor when run repeatedly" do
        rms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn.run xin
          rms_total += nn.rms_error( xtarg )
        end
        rms_total /= 4

        tries = 0
        while ( tries < 10 && ! xor_test(nn) )
          tries += 1
          nn.init_weights
          2000.times do
            xor_train_set.each do | xin, xtarg |
              nn.train_once xin, xtarg
            end
          end
        end

        after_rms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn.run xin
          after_rms_total += nn.rms_error( xtarg )
        end
        after_rms_total /= 4

        after_rms_total.should be < rms_total

        nn.run( NArray.cast( [-1.0, -1.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
        nn.run( NArray.cast( [-1.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn.run( NArray.cast( [1.0, -1.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn.run( NArray.cast( [1.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
      end

      it "can learn xor with 2 hidden layers" do
        rms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn2.run xin
          rms_total += nn2.rms_error( xtarg )
        end
        rms_total /= 4

        tries = 0
        while ( tries < 10 && ! xor_test(nn2) )
          tries += 1
          nn2.init_weights
          4000.times do
            xor_train_set.each do | xin, xtarg |
              nn2.train_once xin, xtarg
            end
          end
        end

        after_rms_total = 0.0
        xor_train_set.each do | xin, xtarg |
          nn2.run xin
          after_rms_total += nn2.rms_error( xtarg )
        end
        after_rms_total /= 4

        after_rms_total.should be < rms_total

        nn2.run( NArray.cast( [-1.0, -1.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
        nn2.run( NArray.cast( [-1.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn2.run( NArray.cast( [1.0, -1.0], 'sfloat' ) )[0].should be_within(0.1).of 1.0
        nn2.run( NArray.cast( [1.0, 1.0], 'sfloat' ) )[0].should be_within(0.1).of 0.0
      end
    end
  end
end
