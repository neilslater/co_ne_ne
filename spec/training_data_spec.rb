require 'helpers'

describe CoNeNe::TrainingData do
  let(:xor_inputs) { NArray.cast( [ [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0] ], 'sfloat' ) }
  let(:xor_targets) { NArray.cast( [ [0.0], [1.0], [1.0], [0.0] ], 'sfloat' ) }

  describe "class methods" do
    describe "#new" do
      it "creates a new object" do
        CoNeNe::TrainingData.new( xor_inputs, xor_targets ).should be_a CoNeNe::TrainingData
      end

      it "should create training data with properties derived from supplied arrays" do
        training = CoNeNe::TrainingData.new( xor_inputs, xor_targets )
        training.inputs.should be xor_inputs
        training.outputs.should be xor_targets
        training.num_items.should == 4
      end

      it "should create training data when input has 3 or more ranks" do
        quad_xor_inputs = NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' )
        training = CoNeNe::TrainingData.new( quad_xor_inputs, xor_targets )
        training.inputs.should be quad_xor_inputs
        training.outputs.should be xor_targets
        training.num_items.should == 4
      end

      it "refuses to create new object when inputs or targets rank is too low" do
        bad_inputs = NArray.cast( [ -1.0, 0.0, 0.5, 1.0 ], 'sfloat' )
        bad_targets = NArray.cast( [ 0.0, 1.0, 1.0, 0.0 ], 'sfloat' )
        expect { CoNeNe::TrainingData.new( bad_inputs, xor_targets ) }.to raise_error
        expect { CoNeNe::TrainingData.new( xor_inputs, bad_targets ) }.to raise_error
      end

      it "refuses to create new object when inputs and targets last dimension does not match" do
        xor_target_missing = NArray.cast( [ [0.0], [1.0], [1.0] ], 'sfloat' )
        expect { CoNeNe::TrainingData.new( xor_inputs, xor_target_missing ) }.to raise_error
      end
    end
  end
end
