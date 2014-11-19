class RuNeNe::MLP::Network

  # Creates new network from an array of layers. The layers are connected together to form the
  # new network - this requires that they have matching input and output sizes on each
  # connection, and they will be reconnected if neccessary to form the new network.
  # @param [Array<RuNeNe::MLP::Layer>] layers an array of layers
  # @return [RuNeNe::MLP::Network] the new network
  def self.from_layers layers
    unless layers.is_a?( Array ) && layers.count > 0 && layers.all? { |l| l.is_a?( RuNeNe::MLP::Layer ) }
      raise TypeError, "Expecting an Array with one or more RuNeNe::MLP::Layer objects"
    end

    # Pre-check for size mismatches
    i = 0
    layers.each_cons(2) do |first,second|
      if first.num_outputs != second.num_inputs
        raise ArgumentError, "Layer #{i} has #{first.num_outputs} outputs, layer #{i+1} has #{second.num_inputs} inputs, they cannot be connected."
      end
      i += 1
    end

    # Force first layer to detach any previous input
    layers[0].set_input( NArray.sfloat( layers[0].num_inputs ) )

    layers.each_cons(2) do |first,second|
      second.attach_input_layer( first )
    end

    from_layer( layers[0] )
  end

end

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

class RuNeNe::MLP::Layer
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
  # @return [RuNeNe::MLP::Layer] new object
  def self.from_h h
    RuNeNe::MLP::Layer.from_weights( h[:weights], h[:transfer] )
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

class RuNeNe::MLP::Network
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
  # @return [RuNeNe::MLP::Network] new object
  def self.from_h h
    hashed_layers = h[:layers]
    restored_layers = hashed_layers.map { |lhash| RuNeNe::MLP::Layer.from_h( lhash ) }
    network = RuNeNe::MLP::Network.from_layers( restored_layers )
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
  # @return [RuNeNe::MLP::Layer] new object
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
