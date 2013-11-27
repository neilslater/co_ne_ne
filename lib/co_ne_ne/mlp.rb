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
