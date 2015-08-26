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

          # For consistency, adding here ensures initial weights are set same each time
          RuNeNe.srand( 893 )
          NArray.srand( 903)
          srand(52) # Needed for :softmax target_type

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

describe "Backprop gradients per layer" do
  for_all_valid_layer_builds do |layer, trainer, objective_type|
    transfer_type = layer.transfer.label
    describe "for FeedForward(#{layer.num_inputs}, #{layer.num_outputs}, #{transfer_type}) and objective #{objective_type}" do
      before :each do
        @inputs = random_inputs( layer.num_inputs )
        @targets = random_targets( layer.num_outputs, transfer_type )

        # This ensures that there is a gradient worth calculating in mlogloss examples (all zeroes is
        # otherwise possible for targets, which is always 0 loss and 0 gradient under mlogloss)
        if ( objective_type == :mlogloss )
          @targets = random_targets( layer.num_outputs, :softmax )
        end

        @outputs = layer.run( @inputs )
        o = objective_module( objective_type )
        @loss_fn = ->(outputs,targets) { o.loss(outputs,targets) }
      end

      it "calculates same de_dz gradients in output layer as RuNeNe::Objective.de_dz" do
        expected_de_dz = RuNeNe::Objective.de_dz( objective_type, transfer_type, @outputs, @targets)
        trainer.start_batch
        trainer.backprop_for_output_layer( layer, @inputs, @outputs, @targets, objective_type )
        expect( trainer.de_dz ).to be_narray_like expected_de_dz
      end

      it "matches measured de_dw gradients in output layer" do
        expected_de_dw = measure_output_layer_de_dw( layer, @loss_fn, @inputs, @targets )
        trainer.start_batch
        trainer.backprop_for_output_layer( layer, @inputs, @outputs, @targets, objective_type )
        expect( trainer.de_dw ).to be_narray_like( expected_de_dw, 1e-7 )
      end

      it "accumulates de_dw gradients in output layer when called twice" do
        expected_de_dw = measure_output_layer_de_dw( layer, @loss_fn, @inputs, @targets )
        trainer.start_batch
        trainer.backprop_for_output_layer( layer, @inputs, @outputs, @targets, objective_type )
        trainer.backprop_for_output_layer( layer, @inputs, @outputs, @targets, objective_type )
        expect( trainer.de_dw ).to be_narray_like( expected_de_dw * 2, 1e-7 )
      end

      it "matches measured de_da gradients from inputs to final layer" do
        expected_de_da = measure_output_layer_de_da( layer, @loss_fn, @inputs, @targets )
        trainer.start_batch
        trainer.backprop_for_output_layer( layer, @inputs, @outputs, @targets, objective_type )
        expect( trainer.de_da ).to be_narray_like( expected_de_da, 1e-7 )
      end
    end
  end
end