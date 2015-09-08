require 'helpers'

describe RuNeNe do
  describe "#weight_decay" do
    before :each do
      @weights = NArray.cast( [ [0.5,-0.3,1.5], [-0.4,0.7,-2.5] ], 'sfloat' )
      @de_dw = NArray.cast( [ [0.1,0.2,-0.3], [-0.4,-0.5,0.6] ], 'sfloat' )
    end

    it "has no effect if decay = 0.0" do
      RuNeNe.weight_decay( @weights, @de_dw, 0.0 )
      expect( @weights ).to be_narray_like NArray[ [0.5,-0.3,1.5], [-0.4,0.7,-2.5] ]
      expect( @de_dw ).to be_narray_like NArray[ [0.1,0.2,-0.3], [-0.4,-0.5,0.6] ]
    end

    it "alters de_dw if decay = 0.01" do
      RuNeNe.weight_decay( @weights, @de_dw, 0.01 )
      expect( @weights ).to be_narray_like NArray[ [0.5,-0.3,1.5], [-0.4,0.7,-2.5] ]
      expect( @de_dw ).to be_narray_like NArray[ [ 0.105, 0.197, -0.3 ], [ -0.404, -0.493, 0.6 ] ]
    end
  end

  describe "#max_norm" do
    before :each do
      @weights = NArray.cast( [ [0.5,-0.3,1.5], [-0.4,0.7,-2.5] ], 'sfloat' )
    end

    it "has no effect if max_norm param is large" do
      RuNeNe.max_norm( @weights, 10.0 )
      expect( @weights ).to be_narray_like NArray[ [0.5,-0.3,1.5], [-0.4,0.7,-2.5] ]
    end

    it "reduces weights if max_norm is smaller than target size" do
      RuNeNe.max_norm( @weights, 0.5 )
      expect( @weights ).to be_narray_like NArray[ [ 0.428746, -0.257248, 1.5 ], [ -0.248069, 0.434122, -2.5 ] ]
    end
  end
end
