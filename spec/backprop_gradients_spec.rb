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
          RuNeNe.srand( 824 )
          NArray.srand( 906 )
          srand(5434) # Needed for :softmax target_type

          layer = RuNeNe::Layer::FeedForward.new( input_size, output_size, transfer_type )
          trainer = RuNeNe::Trainer::BPLayer.from_layer( layer )
          yield layer, trainer, objective_type
        end
      end
    end
  end
end

def for_all_test_networks
  (2..5).each do |input_size|
    (2..4).each do |hidden_size|
      [:linear,:relu,:sigmoid,:tanh,:softmax].each do |hidden_transfer_type|
        (1..3).each do |output_size|
          [:linear,:sigmoid,:softmax].each do |output_transfer_type|
            next if ( output_transfer_type == :softmax && output_size < 2)

            objective = case output_transfer_type
            when :linear then RuNeNe::Objective::MeanSquaredError
            when :sigmoid then RuNeNe::Objective::LogLoss
            when :softmax then RuNeNe::Objective::MulticlassLogLoss
            else raise "Unknown objective type for #{output_transfer_type}"
            end

            # For consistency, adding here ensures initial weights are set same each time
            RuNeNe.srand( 21143 )
            NArray.srand( 98189)
            srand( 3141 ) # Needed for :softmax target_type

            description = "[#{input_size},#{hidden_size}/#{hidden_transfer_type},#{output_size}/#{output_transfer_type}] network"

            nn = TestLayerStack.new( input_size,
              [ [hidden_size,hidden_transfer_type], [output_size,output_transfer_type] ],
              objective
              )
            yield nn, description
          end
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

  for_all_test_networks do |nn,description|
    describe "for #{description}" do
      it "matches measured de_dw gradients in all layers" do
        inputs = random_inputs( nn.layers.first.num_inputs )
        targets = random_targets( nn.layers.last.num_outputs, nn.layers.last.transfer.label )
        nn.start_batch
        nn.process_example(inputs, targets)
        got_de_dws = nn.training_layers.map { |tl| tl.de_dw }
        measured_de_dws = nn.measure_de_dw( inputs, targets )

        got_de_dws.zip( measured_de_dws ).each do |got_de_dw,expect_de_dw|
          expect( got_de_dw ).to be_narray_like( expect_de_dw, 1e-7 )
        end
      end
    end
  end
end
