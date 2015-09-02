# RuNeNe adds support for Marshal to NArray. Code originally from http://blade.nagaokaut.ac.jp/cgi-bin/scat.rb/ruby/ruby-talk/194510
class NArray
  # @!visibility private
  # Adds support for Marshal, via to_h and from_h methods
  def _dump *ignored
    Marshal.dump :typecode => typecode, :shape => shape, :data => to_s
  end

  # @!visibility private
  def self._load buf
    h = Marshal.load buf
    typecode = h[:typecode]
    shape = h[:shape]
    data = h[:data]
    to_na data, typecode, *shape
  end
end

module RuNeNe::Transfer::Sigmoid
  # Short name for Sigmoid transfer function, used as a parameter to some methods.
  # @return [Symbol] :sigmoid
  def self.label
    :sigmoid
  end
end

module RuNeNe::Transfer::TanH
  # Short name for TanH transfer function, used as a parameter to some methods.
  # @return [Symbol] :tanh
  def self.label
    :tanh
  end
end

module RuNeNe::Transfer::ReLU
  # Short name for ReLU transfer function, used as a parameter to some methods.
  # @return [Symbol] :relu
  def self.label
    :relu
  end
end


module RuNeNe::Transfer::Linear
  # Short name for Linear transfer function, used as a parameter to some methods.
  # @return [Symbol] :linear
  def self.label
    :linear
  end
end

module RuNeNe::Transfer::Softmax
  # Short name for Softmax transfer function, used as a parameter to some methods.
  # @return [Symbol] :softmax
  def self.label
    :softmax
  end
end

module RuNeNe::Objective::MeanSquaredError
  # Short name for MeanSquaredError objective function, used as a parameter to some methods.
  # @return [Symbol] :mse
  def self.label
    :mse
  end
end

module RuNeNe::Objective::LogLoss
  # Short name for LogLoss objective function, used as a parameter to some methods.
  # @return [Symbol] :logloss
  def self.label
    :logloss
  end
end

module RuNeNe::Objective::MulticlassLogLoss
  # Short name for MulticlassLogLoss objective function, used as a parameter to some methods.
  # @return [Symbol] :mlogloss
  def self.label
    :mlogloss
  end
end

class RuNeNe::Layer::FeedForward
  # @!visibility private
  # Adds support for Marshal, via to_h and from_h methods
  def to_h
    Hash[
      :weights => self.weights,
      :transfer => self.transfer.label,
    ]
  end

  # @!visibility private
  # Constructs a Layer from hash description. Used internally to support Marshal.
  # @param [Hash] h Keys are :weights and :transfer
  # @return [RuNeNe::Layer::FeedForward] new object
  def self.from_h h
    RuNeNe::Layer::FeedForward.from_weights( h[:weights], h[:transfer] )
  end

  # @!visibility private
  def _dump *ignored
    Marshal.dump to_h
  end

  # @!visibility private
  def self._load buf
    h = Marshal.load buf
    from_h h
  end
end

class RuNeNe::DataSet
  # @!visibility private
  # Adds support for Marshal, via to_h and from_h methods
  def to_h
    Hash[
      :inputs => self.inputs,
      :outputs => self.outputs,
    ]
  end

  # @!visibility private
  # Constructs a DataSet from hash description. Used internally to support Marshal.
  # @param [Hash] h Keys are :weights and :transfer
  # @return [RuNeNe::Layer::FeedForward] new object
  def self.from_h h
    RuNeNe::DataSet.new( h[:inputs], h[:outputs] )
  end

  # @!visibility private
  def _dump *ignored
    Marshal.dump to_h
  end

  # @!visibility private
  def self._load buf
    h = Marshal.load buf
    from_h h
  end
end

class RuNeNe::Learn::MBGD::Layer
  # @!visibility private
  # Adds support for Marshal, via to_h and from_h methods
  def to_h
    Hash[
      [:num_inputs, :num_outputs, :learning_rate, :gd_accel_rate, :weight_decay, :max_norm,
       :gd_accel_type, :de_dz, :de_da, :de_dw, :de_dw_stats_a, :de_dw_stats_b].map do |prop|
        [ prop, self.send(prop) ]
      end
    ]
  end

  # @!visibility private
  # Constructs a MBGD::Layerfrom hash description. Used internally to support Marshal.
  # @param [Hash] h Keys are :weights and :transfer
  # @return [RuNeNe::Learn::MBGD::Layer] new object
  def self.from_h h
    RuNeNe::Learn::MBGD::Layer.new( h )
  end

  # @!visibility private
  def _dump *ignored
    Marshal.dump to_h
  end

  # @!visibility private
  def self._load buf
    h = Marshal.load buf
    from_h h
  end
end