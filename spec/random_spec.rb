require 'helpers'

describe CoNeNe do
  describe "random number generator" do
    it "does not use the default mt.c seed when loaded" do
      # Run in a separate process in order to get library loaded with initial state
      script_output = `ruby -Ilib -e "require 'co_ne_ne'; puts CoNeNe.rand"`
      got_num = script_output.chomp.to_f
      got_num.should_not be_within(1e-8).of 0.3147237002849579
    end

    it "generates numbers consistently when seeded" do
      inputs = [
        [     0, [0.048813, 0.092844, 0.215189, 0.344266, 0.102763, 0.357945, 0.044883, 0.347252, 0.923654] ],
        [   830, [0.958232, 0.129571, 0.470219, 0.028181, 0.131809, 0.675169, 0.473927, 0.302314, 0.488483] ],
        [  7685, [0.608384, 0.474372, 0.841602, 0.002379, 0.250629, 0.524337, 0.009315, 0.591537, 0.874626] ],
        [  7684, [0.639186, 0.009604, 0.583700, 0.239459, 0.475690, 0.691767, 0.598912, 0.521072, 0.958775] ],
      ]

      inputs.each do |seed, expected_results|
        CoNeNe.srand( seed )
        got_results = expected_results.map { |e| CoNeNe.rand }
        got_results.zip( expected_results ).each do |got_val, expected_val|
          got_val.should be_within(1e-6).of expected_val
        end
      end
    end

  end
end
