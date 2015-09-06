require 'helpers'

describe RuNeNe::GradientDescent::SGD do
  describe "class methods" do
    let(:example_params) { NArray.sfloat(2) }

    describe "#new" do
      it "creates a new object" do
        expect( RuNeNe::GradientDescent::SGD.new( example_params ) ).to be_a RuNeNe::GradientDescent::SGD
      end

      it "should take number of params from example input" do
        gd = RuNeNe::GradientDescent::SGD.new( example_params )
        expect( gd.num_params ).to be 2
      end

      it "should take number of params regardless of rank" do
        gd = RuNeNe::GradientDescent::SGD.new( NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' ) )
        expect( gd.num_params ).to be 16
      end
    end

    describe "with Marshal" do
      before do
        @orig_data = RuNeNe::GradientDescent::SGD.new( example_params )
        @saved_data = Marshal.dump( @orig_data )
        @copy_data =  Marshal.load( @saved_data )
      end

      it "can save and retrieve gradient descent settings" do
        expect( @copy_data ).to_not be @orig_data
        expect( @copy_data.num_params ).to be 2
      end
    end
  end
end
