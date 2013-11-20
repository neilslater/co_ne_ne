require 'helpers'

describe CoNeNe::MLP::ZNetwork do
  describe "class methods" do
    describe "#new" do
      it "creates a new network" do
        CoNeNe::MLP::ZNetwork.new( 2, [], 1 ).should be_a CoNeNe::MLP::ZNetwork
        CoNeNe::MLP::ZNetwork.new( 2, [4], 1 ).should be_a CoNeNe::MLP::ZNetwork
        CoNeNe::MLP::ZNetwork.new( 2, [4,2], 1 ).should be_a CoNeNe::MLP::ZNetwork
      end
      it "does not create a new network if any params are missing or bad" do
        expect { CoNeNe::MLP::ZNetwork.new( -2, [4], 1 ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( nil, [4], 1 ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( "a fish", [4], 1 ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( [4], 1 ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( 2, 3, 1 ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( 2, ["z"], 1 ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( 2, [-3], 1 ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( 2, [3,4], -81 ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( 2, [3] ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( 2, [3], nil ) }.to raise_error
        expect { CoNeNe::MLP::ZNetwork.new( 2, [3], 'a frog' ) }.to raise_error
      end
    end
  end
end
