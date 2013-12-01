require 'helpers'

describe CoNeNe::Net::Training do
  let(:xor_inputs) { NArray.cast( [ [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0] ], 'sfloat' ) }
  let(:xor_targets) { NArray.cast( [ [0.0], [1.0], [1.0], [0.0] ], 'sfloat' ) }

  describe "class methods" do
    describe "#new" do
      it "creates a new training item" do
        CoNeNe::Net::Training.new( xor_inputs, xor_targets ).should be_a CoNeNe::Net::Training
      end
    end
  end
end
