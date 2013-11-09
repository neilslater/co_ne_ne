module CoNeNe::Transfer

  module TanH
    def self.function x
      ( 2.0 / (1.0 + Math.exp(-2*x)) ) - 1.0
    end

    def self.bulk_apply_function narr
      narr.collect! { |x| ( 2.0 / (1.0 + Math.exp(-2*x)) ) - 1.0 }
    end

    def self.derivative x
      derivative_at( function( x ) )
    end

    def self.derivative_at y
      1.0 - y * y
    end
  end

  module ReLU
    def self.function x
      x > 0.0 ? x : 0.0
    end

    def self.bulk_apply_function narr
      narr.collect! { |x| x > 0.0 ? x : 0.0 }
    end

    def self.derivative x
      x > 0.0 ? 1.0 : 0.0
    end

    def self.derivative_at y
      y > 0.0 ? 1.0 : 0.0
    end
  end

end
