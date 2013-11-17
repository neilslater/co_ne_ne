module CoNeNe::MLP

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
      @layer_sizes << nil

      @layers = []
      prev_layer = nil
      @layer_sizes.each_cons(3) do | i, o, n |
        transfer_module = n ? :tanh : :sigmoid
        new_layer = CoNeNe::MLP::Layer.new( i, o, transfer_module )
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
      @layers.first.set_input( input )
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
      @layers.last.ms_error target
    end

    def rms_error target
      @layers.last.ms_error target
    end

    def init_weights
      @layers.each { |layer| layer.init_weights }
    end
  end

end
