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

      it "works with different input shapes" do
        shapes = [ [3,3], [7,7,19], [1,1], [12,3], [3,3,3,3] ]
        shapes.each do |ishape|
          oshape = [1,ishape[-1]]
          inputs = NArray.sfloat( *ishape ).random
          outputs = NArray.sfloat( *oshape ).random
          td = RuNeNe::TrainingData.new( inputs, outputs )
          cii = td.current_input_item
          expect( cii ).to be_a NArray
          expect( cii.shape ).to eql ishape[0..(ishape.size-2)]
          expect( inputs.to_a ).to include cii.to_a
        end
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

      it "works with different output shapes" do
        shapes = [ [3,3], [7,7,19], [1,1], [12,3], [3,3,3,3] ]
        shapes.each do |oshape|
          ishape = [4,oshape[-1]]
          inputs = NArray.sfloat( *ishape ).random
          outputs = NArray.sfloat( *oshape ).random
          td = RuNeNe::TrainingData.new( inputs, outputs )
          coi = td.current_output_item
          expect( coi ).to be_a NArray
          expect( coi.shape ).to eql oshape[0..(oshape.size-2)]
          expect( outputs.to_a ).to include coi.to_a
        end
      end
    end

    describe "#next_item" do
      it "changes the current input item" do
        @tdata.next_item
        cii_a = @tdata.current_input_item
        @tdata.next_item
        cii_b = @tdata.current_input_item
        expect( cii_a.to_a ).to_not eql cii_b.to_a
      end

      it "always changes input and output to same item" do
        100.times do
          @tdata.next_item
          cii = @tdata.current_input_item
          coi = @tdata.current_output_item
          xor_result = cii[0] == cii[1] ? 0.0 : 1.0
          expect( coi[0] ).to eql xor_result
        end
      end

      it "shuffles items at the start of each group" do
        # Check single iteration through group
        first_group = (0..3).map do |x|
          @tdata.next_item
          @tdata.current_input_item.to_a
        end
        expect( first_group.uniq ).to eql first_group

        second_group = (0..3).map do |x|
          @tdata.next_item
          @tdata.current_input_item.to_a
        end
        first_group.each do |item|
          expect( second_group ).to include item
        end

        # Check larger groups for differences to reduce likelihood that test fails
        # due to coincidence
        first_group = (0..11).map do |x|
          @tdata.next_item
          @tdata.current_input_item.to_a
        end
        second_group = (0..11).map do |x|
          @tdata.next_item
          @tdata.current_input_item.to_a
        end
        expect( first_group ).to_not eql second_group
      end
    end
  end
end
