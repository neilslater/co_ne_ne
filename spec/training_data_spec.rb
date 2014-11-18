require 'helpers'

describe CoNeNe::TrainingData do
  let(:xor_inputs) { NArray.cast( [ [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0] ], 'sfloat' ) }
  let(:xor_targets) { NArray.cast( [ [0.0], [1.0], [1.0], [0.0] ], 'sfloat' ) }

  describe "class methods" do
    describe "#new" do
      it "creates a new object" do
        expect( CoNeNe::TrainingData.new( xor_inputs, xor_targets ) ).to be_a CoNeNe::TrainingData
      end

      it "should create training data with properties derived from supplied arrays" do
        training = CoNeNe::TrainingData.new( xor_inputs, xor_targets )
        expect( training.inputs ).to be xor_inputs
        expect( training.outputs ).to be xor_targets
        expect( training.num_items ).to be 4
      end

      it "should create training data when input has 3 or more ranks" do
        quad_xor_inputs = NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' )
        training = CoNeNe::TrainingData.new( quad_xor_inputs, xor_targets )
        expect( training.inputs ).to be quad_xor_inputs
        expect( training.outputs ).to be xor_targets
        expect( training.num_items ).to be 4
      end

      it "refuses to create new object when inputs or targets rank is too low" do
        bad_inputs = NArray.cast( [ -1.0, 0.0, 0.5, 1.0 ], 'sfloat' )
        bad_targets = NArray.cast( [ 0.0, 1.0, 1.0, 0.0 ], 'sfloat' )
        expect { CoNeNe::TrainingData.new( bad_inputs, xor_targets ) }.to raise_error ArgumentError
        expect { CoNeNe::TrainingData.new( xor_inputs, bad_targets ) }.to raise_error ArgumentError
      end

      it "refuses to create new object when inputs and targets last dimension does not match" do
        xor_target_missing = NArray.cast( [ [0.0], [1.0], [1.0] ], 'sfloat' )
        expect {
          CoNeNe::TrainingData.new( xor_inputs, xor_target_missing )
        }.to raise_error ArgumentError
      end
    end

    describe "with Marshal" do
      before do
        @orig_data = CoNeNe::TrainingData.new( xor_inputs, xor_targets )
        @saved_data = Marshal.dump( @orig_data )
        @copy_data =  Marshal.load( @saved_data )
      end

      it "can save and retrieve training data" do
        expect( @copy_data ).to_not be @orig_data
        expect( @copy_data.num_items ).to be 4
        orig_inputs = @orig_data.inputs
        copy_inputs = @copy_data.inputs
        expect( copy_inputs ).to_not be orig_inputs
        expect( copy_inputs ).to be_narray_like orig_inputs
        orig_outputs = @orig_data.outputs
        copy_outputs = @copy_data.outputs
        expect( copy_outputs ).to_not be orig_outputs
        expect( copy_outputs ).to be_narray_like orig_outputs
      end
    end
  end
end
