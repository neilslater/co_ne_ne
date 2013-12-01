require 'helpers'

describe CoNeNe::Net::Training do
  describe "class methods" do
    describe "#new" do
      it "creates a new training item" do
        CoNeNe::Net::Training.new( 2, 1 ).should be_a CoNeNe::Net::Training
      end
    end
  end
end
