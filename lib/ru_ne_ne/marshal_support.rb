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

class RuNeNe::NetworkOld
  # @!visibility private
  # Adds support for Marshal, via to_h and from_h methods
  def to_h
    Hash[
      :layers => self.layers.map { |l| l.to_h },
      :lr => self.learning_rate,
      :momentum => self.momentum,
    ]
  end

  # @!visibility private
  # Constructs a Layer from hash description. Used internally to support Marshal.
  # @param [Hash] h Keys are :layers, :lr and :momentum
  # @return [RuNeNe::Network] new object
  def self.from_h h
    hashed_layers = h[:layers]
    restored_layers = hashed_layers.map { |lhash| RuNeNe::Layer::FeedForward.from_h( lhash ) }
    network = RuNeNe::Network.from_layers( restored_layers )
    network.learning_rate = h[:lr]
    network.momentum = h[:momentum]
    network
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

class RuNeNe::TrainingData
  # @!visibility private
  # Adds support for Marshal, via to_h and from_h methods
  def to_h
    Hash[
      :inputs => self.inputs,
      :outputs => self.outputs,
    ]
  end

  # @!visibility private
  # Constructs a TrainingData from hash description. Used internally to support Marshal.
  # @param [Hash] h Keys are :weights and :transfer
  # @return [RuNeNe::Layer::FeedForward] new object
  def self.from_h h
    RuNeNe::TrainingData.new( h[:inputs], h[:outputs] )
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
