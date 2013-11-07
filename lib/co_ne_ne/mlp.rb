module CoNeNe::MLP

  def self.transfer activation
    return 1.0 / (1.0 + Math.exp(-activation))
  end

  def self.transfer_derivative output
    return output * (1.0 - output)
  end

  class Layer
    attr_reader :num_inputs, :num_outputs, :input, :output, :weights
    attr_reader :input_layer, :output_layer, :output_deltas, :weights_last_deltas

    def initialize n_inputs, n_outputs
      @num_inputs = Integer( n_inputs )
      @num_outputs = Integer( n_outputs )
      @output = NArray.sfloat( @num_outputs )
      @weights = NArray.sfloat( @num_inputs + 1,  @num_outputs ).random( 1.54 ) - 0.77
      @weights_last_deltas = NArray.sfloat( @num_inputs + 1,  @num_outputs )
    end

    def self.from_weights new_weights
      if ! new_weights.is_a?(NArray) || new_weights.rank != 2
        raise "Weights array #{new_weights.inspect} unexpected"
      end
      x, y = new_weights.shape
      layer = CoNeNe::MLP::Layer.new( x - 1, y )
      layer.instance_variable_set( :@weights, new_weights )
      layer
    end

    def attach_input new_input
      if ! new_input.is_a?(NArray) || new_input.size != num_inputs || new_input.shape[0] != num_inputs
        raise "Input array #{new_input.inspect} unexpected"
      end
      @input = new_input
      if @input_layer
        @input_layer.instance_variable_set( :@output_layer, nil )
        @input_layer = nil
      end
    end

    def attach_input_layer new_input_layer
      if ! new_input_layer.is_a?(CoNeNe::MLP::Layer) || new_input_layer.num_outputs != num_inputs
        raise "Input layer #{new_input_layer.inspect} unexpected"
      end
      if output_chain.any? { |l| l.object_id == new_input_layer.object_id }
        raise "Attempted cyclic connection"
      end

      if @input_layer
        @input_layer.instance_variable_set( :@output_layer, nil )
      end

      @input_layer = new_input_layer
      @input = new_input_layer.output
      new_input_layer.instance_variable_set( :@output_layer, self )
    end

    def attach_output_layer new_output_layer
      if ! new_output_layer.is_a?(CoNeNe::MLP::Layer) || new_output_layer.num_inputs != num_outputs
        raise "Output layer #{new_output_layer.inspect} unexpected"
      end
      if input_chain.any? { |l| l.object_id == new_output_layer.object_id }
        raise "Attempted cyclic connection"
      end

      if @output_layer
        @output_layer.instance_variable_set( :@input_layer, nil )
        @output_layer.instance_variable_set( :@input, nil )
      end
      @output_layer = new_output_layer
      new_output_layer.attach_input_layer( self )
    end

    def run
      raise "No input!" unless @input
      num_outputs.times do |j|
        activation = (weights[(0...@num_inputs),j] * @input).sum + weights[@num_inputs,j]
        @output[j] = CoNeNe::MLP.transfer( activation )
      end
    end

    def calc_output_deltas target
      if ! target.is_a?(NArray) || target.size != num_outputs || target.shape[0] != @num_outputs
        raise "Target array #{target.inspect} unexpected"
      end
      @output_deltas = NArray.sfloat( @num_outputs ) unless @output_deltas
      @num_outputs.times do |j|
        @output_deltas[j] = ( target[j] - @output[j] ) * CoNeNe::MLP.transfer_derivative( @output[j] )
      end
      @output_deltas
    end

    def rms_error target
      if ! target.is_a?(NArray) || target.size != num_outputs || target.shape[0] != @num_outputs
        raise "Target array #{target.inspect} unexpected"
      end
      diff = target - @output
      ( diff * diff ).sum / @num_outputs
    end

    def backprop_deltas
      raise "No output deltas!" unless @output_deltas
      raise "No input layer!" unless @input_layer
      unless @input_layer.output_deltas
        @input_layer.instance_variable_set( :@output_deltas, NArray.sfloat( @num_inputs ) )
      end
      deltas = @input_layer.output_deltas
      @num_inputs.times do |i|
        deltas[i] = 0.0
        @num_outputs.times do |j|
          deltas[i] += @weights[i,j] * @output_deltas[j]
        end
        deltas[i] *=  CoNeNe::MLP.transfer_derivative( @input[i] )
      end
      deltas
    end

    def update_weights learning_rate, momentum = 0.0
      raise "No output deltas!" unless @output_deltas
      raise "No input!" unless @input

      @num_outputs.times do |j|
        @num_inputs.times do |i|
          wupdate = (learning_rate * @output_deltas[j] * @input[i]) + momentum * @weights_last_deltas[i,j]
          @weights_last_deltas[i,j] = wupdate
          @weights[i,j] += wupdate
        end
        wupdate = (learning_rate * @output_deltas[j]) + momentum * @weights_last_deltas[@num_inputs,j]
        @weights_last_deltas[@num_inputs,j] = wupdate
        @weights[@num_inputs,j] += learning_rate * @output_deltas[j]
      end
    end

    private

    def output_chain
      current_layer = self
      all_outputs = [ current_layer ]
      while current_layer.output_layer
        current_layer = current_layer.output_layer
        all_outputs << current_layer
      end
      all_outputs
    end

    def input_chain
      current_layer = self
      all_inputs = [ current_layer ]
      while current_layer.input_layer
        current_layer = current_layer.input_layer
        all_inputs << current_layer
      end
      all_inputs
    end

  end

  class Network
    def initialize num_inputs, hidden_layer_sizes, num_outputs
      @num_inputs = Integer( num_inputs )
      raise "Need at least 1 input to be valid" unless @num_inputs > 0

      @num_outputs = Integer( num_outputs )
      raise "Need at least 1 output to be valid" unless @num_outputs > 0

      @layer_sizes = [@num_inputs]
      raise "Hidden layer sizes must be an Array" unless hidden_layer_sizes.is_a?(Array)
      hidden_layer_sizes.each do |ls|
        lsize = Integer( ls )
        raise "Need at least 1 neuron in a layer to be valid" unless lsize > 0
        @layer_sizes << lsize
      end
      @layer_sizes << @num_outputs

      @layers = []
      prev_layer = nil
      @layer_sizes.each_cons(2) do | i, o |
        new_layer = CoNeNe::MLP::Layer.new( i, o )
        @layers << new_layer
        if prev_layer
          new_layer.attach_input_layer prev_layer
        end
        prev_layer = new_layer
      end

      @learning_rate = 1.0
      @momentum = 0.5
    end

    attr_reader :num_inputs, :num_outputs, :layer_sizes, :layers
    attr_accessor :learning_rate, :momentum

    def run input
      @layers.first.attach_input( input )
      @layers.each { |layer| layer.run }
      @layers.last.output
    end

    def output
      @layers.last.output
    end

    def train_once input, target
      run( input )
      @layers.last.calc_output_deltas( target )
      @layers.reverse.each do |layer|
        break unless layer.input_layer
        layer.backprop_deltas
      end
      @layers.each do |layer|
        layer.update_weights( @learning_rate, @momentum )
      end
      @layers.last.rms_error target
    end

    def rms_error target
      @layers.last.rms_error target
    end
  end

end
