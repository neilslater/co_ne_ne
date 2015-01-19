require 'helpers'

describe RuNeNe::TrainingData do
  let(:xor_inputs) { NArray.cast( [ [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0] ], 'sfloat' ) }
  let(:xor_targets) { NArray.cast( [ [0.0], [1.0], [1.0], [0.0] ], 'sfloat' ) }

  describe "class methods" do
    describe "#new" do
      it "creates a new object" do
        expect( RuNeNe::TrainingData.new( xor_inputs, xor_targets ) ).to be_a RuNeNe::TrainingData
      end

      it "should create training data with properties derived from supplied arrays" do
        training = RuNeNe::TrainingData.new( xor_inputs, xor_targets )
        expect( training.inputs ).to be xor_inputs
        expect( training.outputs ).to be xor_targets
        expect( training.num_items ).to be 4
      end

      it "should create training data when input has 3 or more ranks" do
        quad_xor_inputs = NArray.cast( [ [ [-1.0, -1.0], [1.0, -1.0] ],
            [ [-1.0, 1.0], [1.0, 1.0] ], [ [ 1.0, -1.0], [1.0, -1.0] ],
            [ [ 1.0, -1.0], [1.0, -1.0] ] ], 'sfloat' )
        training = RuNeNe::TrainingData.new( quad_xor_inputs, xor_targets )
        expect( training.inputs ).to be quad_xor_inputs
        expect( training.outputs ).to be xor_targets
        expect( training.num_items ).to be 4
      end

      it "refuses to create new object when inputs or targets rank is too low" do
        bad_inputs = NArray.cast( [ -1.0, 0.0, 0.5, 1.0 ], 'sfloat' )
        bad_targets = NArray.cast( [ 0.0, 1.0, 1.0, 0.0 ], 'sfloat' )
        expect { RuNeNe::TrainingData.new( bad_inputs, xor_targets ) }.to raise_error ArgumentError
        expect { RuNeNe::TrainingData.new( xor_inputs, bad_targets ) }.to raise_error ArgumentError
      end

      it "refuses to create new object when inputs and targets last dimension does not match" do
        xor_target_missing = NArray.cast( [ [0.0], [1.0], [1.0] ], 'sfloat' )
        expect {
          RuNeNe::TrainingData.new( xor_inputs, xor_target_missing )
        }.to raise_error ArgumentError
      end
    end

    describe "with Marshal" do
      before do
        @orig_data = RuNeNe::TrainingData.new( xor_inputs, xor_targets )
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

  describe "instance methods" do
    before :each do
      @tdata = RuNeNe::TrainingData.new( xor_inputs, xor_targets )
    end

    describe "#clone" do
      it "makes deep copy of training data" do
        @copy_data = @tdata.clone
        expect( @copy_data ).to_not be @tdata
        expect( @copy_data.num_items ).to be 4
        orig_inputs = @tdata.inputs
        copy_inputs = @copy_data.inputs
        expect( copy_inputs ).to_not be orig_inputs
        expect( copy_inputs ).to be_narray_like orig_inputs
        orig_outputs = @tdata.outputs
        copy_outputs = @copy_data.outputs
        expect( copy_outputs ).to_not be orig_outputs
        expect( copy_outputs ).to be_narray_like orig_outputs
      end
    end

    describe "#current_input_item" do
      it "returns an NArray of correct shape" do
        cii = @tdata.current_input_item
        expect( cii ).to be_a NArray
        expect( cii.shape ).to eql [2]
      end

      it "is a sample from the instance's input data" do
        ciia = @tdata.current_input_item.to_a
        expect( [ [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0] ] ).to include ciia
      end
    end

    describe "#current_output_item" do
      it "returns an NArray of correct shape" do
        coi = @tdata.current_output_item
        expect( coi ).to be_a NArray
        expect( coi.shape ).to eql [1]
      end

      it "is a sample from the instance's output data" do
        coia = @tdata.current_output_item.to_a
        expect( [ [0.0], [1.0], [1.0], [0.0] ] ).to include coia
      end
    end
  end
end
