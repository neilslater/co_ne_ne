class CoNeNe::MLP::Network

  def self.from_layers layers
    unless layers.is_a?( Array ) && layers.count > 0 && layers.all? { |l| l.is_a?( CoNeNe::MLP::Layer ) }
      raise TypeError, "Expecting an Array with one or more CoNeNe::MLP::Layer objects"
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

# This adds support for Marshal to NArray - found it at: http://blade.nagaokaut.ac.jp/cgi-bin/scat.rb/ruby/ruby-talk/194510
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

module CoNeNe::Transfer::Sigmoid
  def self.to_sym
    :sigmoid
  end
end

module CoNeNe::Transfer::TanH
  def self.to_sym
    :tanh
  end
end

module CoNeNe::Transfer::ReLU
  def self.to_sym
    :relu
  end
end

class CoNeNe::MLP::Layer
  # @!visibility private
  # Adds support for Marshal, via to_h and from_h methods
  def to_h
    Hash[
      :weights => self.weights,
      :transfer => self.transfer.to_sym,
    ]
  end

  def self.from_h h
    CoNeNe::MLP::Layer.from_weights( h[:weights], h[:transfer] )
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

class CoNeNe::MLP::Network
  # @!visibility private
  # Adds support for Marshal, via to_h and from_h methods
  def to_h
    Hash[
      :layers => self.layers.map { |l| l.to_h },
    ]
  end

  def self.from_h h
    hashed_layers = h[:layers]
    restored_layers = hashed_layers.map { |lhash| CoNeNe::MLP::Layer.from_h( lhash ) }
    CoNeNe::MLP::Network.from_layers( restored_layers )
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
