require 'helpers'

def for_all_valid_layer_builds
  (2..5).each do |input_size|
    (1..3).each do |output_size|
      [:linear,:relu,:sigmoid,:tanh,:softmax].each do |transfer_type|
        next if ( transfer_type == :softmax && output_size < 2)
        [:mse,:logloss,:mlogloss].each do |objective_type|
          next if ( objective_type == :mlogloss && output_size < 2)
          check_compat = RuNeNe::Objective.de_dz( objective_type, transfer_type, NArray[0.5,0.1], NArray[0.2,0.8]) rescue nil
          next if check_compat.nil?

          # For consistency
          RuNeNe.srand( 893 )
          NArray.srand( 903)

          layer = RuNeNe::Layer::FeedForward.new( input_size, output_size, transfer_type )
          trainer = RuNeNe::Trainer::BPLayer.from_layer( layer )
          yield layer, trainer, objective_type
        end
      end
    end
  end
end

def random_inputs n
  NArray.sfloat(n).random(2.0) - 1.0
end

def random_targets n, target_type
  case target_type
  when :linear then NArray.sfloat(n).random(4.0) - 2.0
  when :relu then NArray.sfloat(n).random(2.0)
  when :tanh then NArray.sfloat(n).random(2.0) - 1.0
  when :sigmoid then NArray.int(n).random(2).to_f
  when :softmax then
    targets = NArray.int(n).to_f
    targets[ rand(n) ] = 1.0
    targets
  end
end

def objective_module objective_type
  case objective_type
  when :mse then RuNeNe::Objective::MeanSquaredError
  when :logloss then RuNeNe::Objective::LogLoss
  when :mlogloss then RuNeNe::Objective::MulticlassLogLoss
  end
end

describe "Layer Gradients" do
  for_all_valid_layer_builds do |layer, trainer, objective_type|
    describe "for FeedForward(#{layer.num_inputs}, #{layer.num_outputs}, #{layer.transfer.label}) and objective #{objective_type}" do
      before :each do
        @inputs = random_inputs( layer.num_inputs )
        @targets = random_targets( layer.num_outputs, layer.transfer.label )

        # This ensures that there is a gradient worth calculating in mlogloss scenarios
        if ( objective_type == :mlogloss )
          @targets = random_targets( layer.num_outputs, :softmax )
        end

        @outputs = layer.run( @inputs )
        o = objective_module( objective_type )
        @loss_fn = ->(outputs,targets) { o.loss(outputs,targets) }
      end

      it "calculates loss" do
        loss = @loss_fn.call(@outputs, @targets)
        expect(loss).to be > 0.0
      end
    end
  end
end