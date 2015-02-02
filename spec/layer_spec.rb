require 'helpers'

describe RuNeNe::Layer::FeedForward do
  describe "class methods" do
    describe "#new" do
      it "creates a new layer" do
        expect( RuNeNe::Layer::FeedForward.new( 2, 1 ) ).to be_a RuNeNe::Layer::FeedForward
      end

      it "refuses to create new layers for bad parameters" do
        expect { RuNeNe::Layer::FeedForward.new( 0, 2 ) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( 3, -1 ) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( "hello", 2 ) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( 3, 2, "garbage" ) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( 3, 2, :foobar ) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( 3, 2, :tanh, 17 ) }.to raise_error
      end

      it "sets values of attributes based on input and output size" do
        RuNeNe.srand( 7000 )

        layer = RuNeNe::Layer::FeedForward.new( 3, 2 )
        expect( layer.num_inputs ).to be 3
        expect( layer.num_outputs ).to be 2
        expect( layer.transfer ).to be RuNeNe::Transfer::Sigmoid

        expect( layer.weights ).to be_narray_like NArray[
         [ 0.586514, 0.637854, 0.525343, 0.718463 ],
         [ 0.957125, 0.0273813, 0.0794556, -0.730958 ] ]
      end

      it "accepts an optional transfer function type param" do
        layer = RuNeNe::Layer::FeedForward.new( 4, 1, :sigmoid )
        expect( layer.transfer ).to be RuNeNe::Transfer::Sigmoid

        layer = RuNeNe::Layer::FeedForward.new( 5, 3, :tanh )
        expect( layer.transfer ).to be RuNeNe::Transfer::TanH

        layer = RuNeNe::Layer::FeedForward.new( 7, 2, :relu )
        expect( layer.transfer ).to be RuNeNe::Transfer::ReLU

        layer = RuNeNe::Layer::FeedForward.new( 17, 1, :linear )
        expect( layer.transfer ).to be RuNeNe::Transfer::Linear
      end
    end

    describe "#from_weights" do
      it "creates a new layer" do
        expect( RuNeNe::Layer::FeedForward.from_weights( NArray.sfloat(4,5) ) ).to be_a RuNeNe::Layer::FeedForward
      end

      it "initialises sizes and output arrays" do
        layer = RuNeNe::Layer::FeedForward.from_weights( NArray.sfloat(4,5) )

        expect( layer.num_inputs ).to be 3
        expect( layer.num_outputs ).to be 5
      end

      it "assigns to the weights attribute directly (not a copy)" do
        w =  NArray.sfloat(12,7)
        layer = RuNeNe::Layer::FeedForward.from_weights( w )
        expect( layer.weights ).to be w
      end

      it "accepts an optional transfer function type param" do
        w =  NArray.sfloat(3,2)
        layer = RuNeNe::Layer::FeedForward.from_weights( w, :sigmoid )
        expect( layer.transfer ).to be RuNeNe::Transfer::Sigmoid

        w =  NArray.sfloat(3,2)
        layer = RuNeNe::Layer::FeedForward.from_weights( w, :tanh )
        expect( layer.transfer ).to be RuNeNe::Transfer::TanH

        w =  NArray.sfloat(3,2)
        layer = RuNeNe::Layer::FeedForward.from_weights( w, :relu )
        expect( layer.transfer ).to be RuNeNe::Transfer::ReLU
      end

      it "refuses to create new layers for bad parameters" do
        expect { RuNeNe::Layer::FeedForward.new( NArray.sfloat(3,2,1) ) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( NArray.sfloat(1,2) ) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( NArray.sfloat(7)) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( NArray.sfloat(5,2), "NOTVALID" ) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( NArray.sfloat(4,1), :blah ) }.to raise_error
        expect { RuNeNe::Layer::FeedForward.new( NArray.sfloat(4,1), :tanh, "extras" ) }.to raise_error
      end
    end

    describe "with Marshal" do
      it "can save and retrieve a layer, preserving weights and transfer function" do
        orig_layer = RuNeNe::Layer::FeedForward.new( 5, 3, :sigmoid )
        saved = Marshal.dump( orig_layer )
        copy_layer = Marshal.load( saved )

        expect( copy_layer ).to_not be orig_layer
        expect( copy_layer.num_inputs ).to be 5
        expect( copy_layer.num_outputs ).to be 3
        expect( copy_layer.transfer ).to be RuNeNe::Transfer::Sigmoid
        expect( copy_layer.weights ).to be_narray_like orig_layer.weights
      end
    end
  end

  describe "instance methods" do
    let :layer do
      weights = NArray.cast( [ [ -0.1, 0.5, 0.9, 0.7 ], [ -0.6, 0.6, 0.4, 0.6 ] ], 'sfloat' )
      RuNeNe::Layer::FeedForward.from_weights( weights )
    end

    describe "#clone" do
      it "should make a simple copy of number of inputs, outputs and transfer function" do
        copy = layer.clone
        expect( copy.num_inputs ).to eql layer.num_inputs
        expect( copy.num_outputs ).to eql layer.num_outputs
        expect( copy.transfer ).to eql layer.transfer
      end

      it "should deep clone weights" do
        copy = layer.clone

        expect( copy.weights ).to_not be layer.weights
        expect( copy.weights ).to be_narray_like layer.weights
      end

      it "should copy the transfer function" do
        layer2 = RuNeNe::Layer::FeedForward.new( 4, 3, :tanh )
        copy = layer2.clone
        expect( copy.transfer ).to be RuNeNe::Transfer::TanH

        layer3 = RuNeNe::Layer::FeedForward.new( 4, 3, :relu )
        copy = layer3.clone
        expect( copy.transfer ).to be RuNeNe::Transfer::ReLU
      end
    end

    describe "#init_weights" do
      before :each do
        RuNeNe.srand(800)
      end

      it "should set weights to normal distribution by default" do
        layer.init_weights
        expect( layer.weights ).to be_narray_like NArray[
          [ 0.26017, -0.128192, -0.184518, 0.129694 ],
          [ -0.366275, 0.188048, -0.0444051, -0.103807 ] ]
      end

      it "should accept an optional multiplier" do
        layer.init_weights( 0.1 )
        expect( layer.weights ).to be_narray_like NArray[
          [ 0.026017, -0.0128192, -0.0184518, 0.0129694 ],
          [ -0.0366275, 0.0188048, -0.00444051, -0.0103807 ] ]
      end

      it "returns self" do
        expect( layer.init_weights() ).to be layer
        expect( layer.init_weights( 2.5 ) ).to be layer
      end
    end

    describe "#run" do
      let(:input) {  NArray.cast( [0.1, 0.2, 0.3], 'sfloat' ) }

      it "calculates output associated with input and weights" do
        output = layer.run( input )
        expect( output ).to be_narray_like NArray[ 0.742691, 0.68568 ]
      end

      it "gives different output for different input" do
        result_one = layer.run( input )

        other_input = NArray.cast( [0.5, 0.4, 0.3], 'sfloat' )
        result_two = layer.run( other_input )

        expect( result_one ).to_not eq result_two
        expect( result_one ).to_not be_narray_like result_two
      end

      it "gives similar output for similar input" do
        result_one = layer.run( input )

        other_input =NArray.cast( [0.1002, 0.1998, 0.3001], 'sfloat' )
        result_two = layer.run( other_input )

        expect( result_one ).to be_narray_like result_two
        expect( result_one ).to_not eq result_two
      end

      it "sets all output values between 0 and 1 (for sigmoid)" do
        output = layer.run( input )
        output.each do |r|
          expect( r ).to be >= 0.0
          expect( r ).to be <= 1.0
        end
      end
    end
  end
end
