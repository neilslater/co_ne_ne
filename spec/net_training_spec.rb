require 'helpers'

describe CoNeNe::Net::Training do
  let(:xor_inputs) { NArray.cast( [ [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0] ], 'sfloat' ) }
  let(:xor_targets) { NArray.cast( [ [0.0], [1.0], [1.0], [0.0] ], 'sfloat' ) }

  describe "class methods" do
    describe "#new" do
      it "creates a new training item" do
        CoNeNe::Net::Training.new( xor_inputs, xor_targets ).should be_a CoNeNe::Net::Training
      end

      it "should create a training item with properties derived from supplied arrays" do
        training = CoNeNe::Net::Training.new( xor_inputs, xor_targets )
      end

      it "should create a training item when input has 3 or more ranks" do
        quad_xor_inputs = NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' )
        training = CoNeNe::Net::Training.new( quad_xor_inputs, xor_targets )
      end

      it "refuses to create new object when inputs or targets rank is too low" do
        bad_inputs = NArray.cast( [ -1.0, 0.0, 0.5, 1.0 ], 'sfloat' )
        bad_targets = NArray.cast( [ 0.0, 1.0, 1.0, 0.0 ], 'sfloat' )
        expect { CoNeNe::Net::Training.new( bad_inputs, xor_targets ) }.to raise_error
        expect { CoNeNe::Net::Training.new( xor_inputs, bad_targets ) }.to raise_error
      end

      it "refuses to create new object when inputs and targets last dimension does not match" do
        xor_target_missing = NArray.cast( [ [0.0], [1.0], [1.0] ], 'sfloat' )
        expect { CoNeNe::Net::Training.new( xor_inputs, xor_target_missing ) }.to raise_error
      end
    end
  end
end
