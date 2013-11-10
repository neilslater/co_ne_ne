module CoNeNe::Transfer

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
