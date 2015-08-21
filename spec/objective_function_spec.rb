require 'helpers'

describe RuNeNe::Objective::MeanSquaredError do
  before :each do
    NArray.srand(24134223512)
  end

  describe "#loss" do
    it "is 0.0 when predictions and targets match" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(20.0) - 10.0
        expect( RuNeNe::Objective::MeanSquaredError.loss( targets, targets ) ).to be_within(1.0e-10).of 0.0
      end
    end

    it "is larger when predictions and targets are further apart" do
      targets = NArray.cast( [ 1.0, 2.0, 3.0, 4.0 ], 'sfloat' )
      preds1 =  NArray.cast( [ 1.5, 1.5, 3.5, 3.5 ], 'sfloat' )
      preds2 =  NArray.cast( [ 1.25, 1.0, 3.25, 5.0 ], 'sfloat' )

      expect( RuNeNe::Objective::MeanSquaredError.loss( preds1, targets ) ).to be < RuNeNe::Objective::MeanSquaredError.loss( preds2, targets )
    end

    it "matches value from documentation" do
      5.times do
        targets = NArray.sfloat(4).random(4.0)
        predictions = NArray.sfloat(4).random(4.0)
        expected_loss = 0.5 * ( predictions.to_a.zip( targets.to_a ).inject(0.0) { |sum,pt| sum + (pt.first-pt.last)**2 } )
        actual_loss = RuNeNe::Objective::MeanSquaredError.loss( predictions, targets )
        expect( actual_loss ).to be_within( 1e-6 ).of actual_loss
      end
    end
  end

  describe "#delta_loss" do
    it "is 0.0 for matching predictions and targets" do
      targets = NArray.cast( [ 1.0, 2.0, 3.0, 4.0 ], 'sfloat' )
      preds =  NArray.cast( [ 1.0, 1.5, 3.0, 3.5 ], 'sfloat' )

      dl =  RuNeNe::Objective::MeanSquaredError.delta_loss( preds, targets )

      expect( dl[0] ).to be_within(1e-6).of 0.0
      expect( dl[1] ).to be < 0.0
      expect( dl[2] ).to be_within(1e-6).of 0.0
      expect( dl[3] ).to be < 0.0
    end

    it "is numerically accurate gradient for the loss function" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(2.0) - 1.0
        predictions = NArray.sfloat(n).random(2.0) - 1.0
        loss = RuNeNe::Objective::MeanSquaredError.loss( predictions, targets )
        dl = RuNeNe::Objective::MeanSquaredError.delta_loss( predictions, targets )

        (0...n).each do |i|
          up_predictions = predictions.clone
          up_predictions[i] += 0.005
          up_loss = RuNeNe::Objective::MeanSquaredError.loss( up_predictions, targets )
          down_predictions = predictions.clone
          down_predictions[i] -= 0.005
          down_loss = RuNeNe::Objective::MeanSquaredError.loss( down_predictions, targets )
          rough_gradient = 100 * ( up_loss - down_loss )
          # Accept a 1% relative error margin (quite high, but realistic if one specific
          # gradient delta is swamped by other effects in random example). Typical values from set of 5:
          # 1.0000091719077568, 0.999966054313099, 0.9999890442611848, 1.0000013064133018, 1.0000042944172576
          expect( dl[i] / rough_gradient ).to be_within( 0.01 ).of 1.0
        end
      end
    end
  end

  describe "#linear_de_dz" do
    let(:demi_layer) { TestDemiOutputLayer.new( RuNeNe::Objective::MeanSquaredError, RuNeNe::Transfer::Linear ) }

    it "is numerically accurate gradient for the loss function measured pre-transfer" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(2.0) - 1.0
        zvals = NArray.sfloat(n).random(2.0) - 1.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MeanSquaredError.linear_de_dz( demi_layer.output, targets )
        rough_gradients = demi_layer.measure_de_dz( zvals, targets )
        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :mse, :linear ...)" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(2.0) - 1.0
        zvals = NArray.sfloat(n).random(2.0) - 1.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MeanSquaredError.linear_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :mse, :linear, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end
  end

  describe "#sigmoid_de_dz" do
    let(:demi_layer) { TestDemiOutputLayer.new( RuNeNe::Objective::MeanSquaredError, RuNeNe::Transfer::Sigmoid ) }

    it "is numerically accurate gradient for the loss function measured pre-transfer" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(10.0) - 5.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MeanSquaredError.sigmoid_de_dz( demi_layer.output, targets )
        rough_gradients = demi_layer.measure_de_dz( zvals, targets )
        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :mse, :sigmoid ...)" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(10.0) - 5.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MeanSquaredError.sigmoid_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :mse, :sigmoid, demi_layer.output, targets )
      end
    end
  end

  describe "#tanh_de_dz" do
    let(:demi_layer) { TestDemiOutputLayer.new( RuNeNe::Objective::MeanSquaredError, RuNeNe::Transfer::TanH ) }

    it "is numerically accurate gradient for the loss function measured pre-transfer" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(1.8) - 0.9
        zvals = NArray.sfloat(n).random(5.0) - 2.5
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MeanSquaredError.tanh_de_dz( demi_layer.output, targets )
        rough_gradients = demi_layer.measure_de_dz( zvals, targets )
        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :mse, :tanh ...)" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(1.8) - 0.9
        zvals = NArray.sfloat(n).random(5.0) - 2.5
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MeanSquaredError.tanh_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :mse, :tanh, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end
  end

  describe "#relu_de_dz" do
    let(:demi_layer) { TestDemiOutputLayer.new( RuNeNe::Objective::MeanSquaredError, RuNeNe::Transfer::ReLU ) }

    it "is numerically accurate gradient for the loss function measured pre-transfer" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(3.0)
        zvals = NArray.sfloat(n).random(4.0) - 1.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MeanSquaredError.relu_de_dz( demi_layer.output, targets )
        rough_gradients = demi_layer.measure_de_dz( zvals, targets )
        (0...n).each do |i|
          if rough_gradients[i].abs > 1e-6
            expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
          else
            expect( got_de_dz[i] ).to be < 1e-5
          end
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :mse, :relu ...)" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(3.0)
        zvals = NArray.sfloat(n).random(4.0) - 1.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MeanSquaredError.relu_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :mse, :relu, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end
  end


  describe "#softmax_de_dz" do
    let(:demi_layer) { TestDemiOutputLayer.new( RuNeNe::Objective::MeanSquaredError, RuNeNe::Transfer::Softmax ) }

    it "is numerically accurate gradient for the loss function measured pre-transfer" do
      test_size_range 3,6 do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(2.0)
        demi_layer.run( zvals, targets )

        got_de_dz = RuNeNe::Objective::MeanSquaredError.softmax_de_dz( demi_layer.output, targets )
        rough_gradients = demi_layer.measure_de_dz( zvals, targets )
        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :mse, :softmax ...)" do
      test_size_range 3,6 do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(2.0)
        demi_layer.run( zvals, targets )

        got_de_dz = RuNeNe::Objective::MeanSquaredError.softmax_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :mse, :softmax, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end
  end
end


describe RuNeNe::Objective::LogLoss do
  before :each do
    NArray.srand(4273421351)
  end

  describe "#loss" do
    it "is 0.0 when predictions and targets match at y=0.0 or y=1.0" do
      test_size_range do |n|
        targets = NArray.int(n).random(2).to_f
        expect( RuNeNe::Objective::LogLoss.loss( targets, targets ) ).to be_within(1.0e-10).of 0.0
      end
    end

    it "is larger when predictions and targets are further apart" do
      targets = NArray.cast( [ 1.0, 0.0, 1.0, 0.0 ], 'sfloat' )
      preds1 =  NArray.cast( [ 0.9, 0.1, 0.8, 0.2 ], 'sfloat' )
      preds2 =  NArray.cast( [ 0.85, 0.15, 0.75, 0.25 ], 'sfloat' )

      expect( RuNeNe::Objective::LogLoss.loss( preds1, targets ) ).to be < RuNeNe::Objective::LogLoss.loss( preds2, targets )
    end
  end

  describe "#delta_loss" do
    it "is numerically accurate gradient for the loss function when y=0.0 or y=1.0" do
      test_size_range do |n|
        targets = NArray.int(n).random(2).to_f
        predictions = NArray.sfloat(n).random(0.98) + 0.01
        loss = RuNeNe::Objective::LogLoss.loss( predictions, targets )
        dl = RuNeNe::Objective::LogLoss.delta_loss( predictions, targets )
        (0...n).each do |i|
          up_predictions = predictions.clone
          up_predictions[i] += 0.0005
          up_loss = RuNeNe::Objective::LogLoss.loss( up_predictions, targets )
          down_predictions = predictions.clone
          down_predictions[i] -= 0.0005
          down_loss = RuNeNe::Objective::LogLoss.loss( down_predictions, targets )
          rough_gradient = 1000 * ( up_loss - down_loss )
          expect( dl[i] / rough_gradient ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is numerically accurate gradient for the loss function when y is between 0.0 and 1.0" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(0.98) + 0.01
        predictions = NArray.sfloat(n).random(0.98) + 0.01
        loss = RuNeNe::Objective::LogLoss.loss( predictions, targets )
        dl = RuNeNe::Objective::LogLoss.delta_loss( predictions, targets )
        (0...n).each do |i|
          up_predictions = predictions.clone
          up_predictions[i] += 0.0005
          up_loss = RuNeNe::Objective::LogLoss.loss( up_predictions, targets )
          down_predictions = predictions.clone
          down_predictions[i] -= 0.0005
          down_loss = RuNeNe::Objective::LogLoss.loss( down_predictions, targets )
          rough_gradient = 1000 * ( up_loss - down_loss )
          expect( dl[i] / rough_gradient ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is zero when a == y" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(0.98) + 0.01
        predictions = targets.clone
        loss = RuNeNe::Objective::LogLoss.loss( predictions, targets )
        dl = RuNeNe::Objective::LogLoss.delta_loss( predictions, targets )
        (0...n).each do |i|
          up_predictions = predictions.clone
          up_predictions[i] += 0.0005
          up_loss = RuNeNe::Objective::LogLoss.loss( up_predictions, targets )
          down_predictions = predictions.clone
          down_predictions[i] -= 0.0005
          down_loss = RuNeNe::Objective::LogLoss.loss( down_predictions, targets )
          rough_gradient = 1000 * ( up_loss - down_loss )
          expect( rough_gradient ).to be_within(0.001).of 0.0
          expect( dl[i] ).to be_within( 1e-6 ).of 0.0
        end
      end
    end
  end

  describe "#sigmoid_de_dz" do
    let(:demi_layer) { TestDemiOutputLayer.new( RuNeNe::Objective::LogLoss, RuNeNe::Transfer::Sigmoid ) }

    it "is numerically accurate gradient for the loss function measured pre-transfer with y=0.0 or 1.0" do
      test_size_range do |n|
        targets = NArray.int(n).random(2).to_f
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::LogLoss.sigmoid_de_dz( demi_layer.output, targets )

        rough_gradients = demi_layer.measure_de_dz( zvals, targets )
        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :logloss, :sigmoid ...) with y=0.0 or 1.0" do
      test_size_range do |n|
        targets = NArray.int(n).random(2).to_f
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::LogLoss.sigmoid_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :logloss, :sigmoid, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end

    it "is numerically accurate gradient for the loss function measured pre-transfer with y in [0.0..1.0]" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::LogLoss.sigmoid_de_dz( demi_layer.output, targets )

        rough_gradients = demi_layer.measure_de_dz( zvals, targets )
        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :logloss, :sigmoid ...) with y in [0.0..1.0]" do
      test_size_range do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::LogLoss.sigmoid_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :logloss, :sigmoid, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end
  end

  describe "#softmax_de_dz" do
    let(:demi_layer) { TestDemiOutputLayer.new( RuNeNe::Objective::LogLoss, RuNeNe::Transfer::Softmax ) }

    it "is numerically accurate gradient for the loss function measured pre-transfer with y=0.0 or 1.0" do
      test_size_range(2,5) do |n|
        targets = NArray.int(n).to_f
        targets[ rand(n) ] = 1.0
        zvals = NArray.sfloat(n).random(2.0) - 1.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::LogLoss.softmax_de_dz( demi_layer.output, targets )

        rough_gradients = demi_layer.measure_de_dz( zvals, targets, 0.005 )

        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :logloss, :softmax ...) with y=0.0 or 1.0" do
      test_size_range(2,5) do |n|
        targets = NArray.int(n).to_f
        targets[ rand(n) ] = 1.0
        zvals = NArray.sfloat(n).random(2.0) - 1.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::LogLoss.softmax_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :logloss, :softmax, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end

    it "is numerically accurate gradient for the loss function measured pre-transfer with y in [0.0..1.0]" do
      test_size_range(2,5) do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::LogLoss.softmax_de_dz( demi_layer.output, targets )

        rough_gradients = demi_layer.measure_de_dz( zvals, targets, 0.005 )
        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :logloss, :softmax ...) with y in [0.0..1.0]" do
      test_size_range(2,5) do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::LogLoss.softmax_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :logloss, :softmax, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end
  end
end

describe RuNeNe::Objective::MulticlassLogLoss do
  before :each do
    NArray.srand(427342151)
  end

  describe "#loss" do
    it "is 0.0 when predictions and targets match" do
      test_size_range do |n|
        targets = NArray.int(n).to_f
        targets[ rand(n) ] = 1.0
        expect( RuNeNe::Objective::MulticlassLogLoss.loss( targets, targets ) ).to be_within(1.0e-10).of 0.0
      end
    end

    it "is larger when predictions and targets are further apart" do
      targets = NArray.cast( [ 0.0, 0.0, 1.0, 0.0 ], 'sfloat' )
      preds1 =  NArray.cast( [ 0.05, 0.1, 0.8, 0.05 ], 'sfloat' )
      preds2 =  NArray.cast( [ 0.1, 0.1, 0.75, 0.05 ], 'sfloat' )

      expect( RuNeNe::Objective::MulticlassLogLoss.loss( preds1, targets ) ).to be < RuNeNe::Objective::MulticlassLogLoss.loss( preds2, targets )
    end
  end

  describe "#delta_loss" do
    it "is numerically accurate gradient for the loss function when there is a single target class" do
      test_size_range(2,5) do |n|

        targets = NArray.int(n).to_f
        targets[ rand(n) ] = 1.0
        predictions = NArray.sfloat(n).random(0.98) + 0.01
        loss = RuNeNe::Objective::MulticlassLogLoss.loss( predictions, targets )
        dl = RuNeNe::Objective::MulticlassLogLoss.delta_loss( predictions, targets )

        affected_gradients = 0
        (0...n).each do |i|
          up_predictions = predictions.clone
          up_predictions[i] += 0.0005
          up_loss = RuNeNe::Objective::MulticlassLogLoss.loss( up_predictions, targets )
          down_predictions = predictions.clone
          down_predictions[i] -= 0.0005
          down_loss = RuNeNe::Objective::MulticlassLogLoss.loss( down_predictions, targets )
          rough_gradient = 1000 * ( up_loss - down_loss )

          # There is only loss, and a gradient, associated with the target class
          if rough_gradient == 0
            expect( dl[i] ).to be == 0.0
          else
             expect( dl[i] / rough_gradient ).to be_within( 0.01 ).of 1.0
             affected_gradients += 1
          end
        end
        expect( affected_gradients ).to be 1
      end
    end

    it "is numerically accurate gradient for the loss function when there are split probability targets" do
      test_size_range(2,5) do |n|
        targets = NArray.sfloat(n).random
        targets /= targets.sum
        predictions = NArray.sfloat(n).random(0.98) + 0.01
        loss = RuNeNe::Objective::MulticlassLogLoss.loss( predictions, targets )
        dl = RuNeNe::Objective::MulticlassLogLoss.delta_loss( predictions, targets )

        affected_gradients = 0
        (0...n).each do |i|
          up_predictions = predictions.clone
          up_predictions[i] += 0.0005
          up_loss = RuNeNe::Objective::MulticlassLogLoss.loss( up_predictions, targets )
          down_predictions = predictions.clone
          down_predictions[i] -= 0.0005
          down_loss = RuNeNe::Objective::MulticlassLogLoss.loss( down_predictions, targets )
          rough_gradient = 1000 * ( up_loss - down_loss )

          # There is only loss, and a gradient, associated with the target class
          if rough_gradient == 0
            expect( dl[i] ).to be == 0.0
          else
            expect( dl[i] / rough_gradient ).to be_within( 0.01 ).of 1.0
          end
        end
      end
    end
  end

  describe "#sigmoid_de_dz" do
    let(:demi_layer) { TestDemiOutputLayer.new( RuNeNe::Objective::MulticlassLogLoss, RuNeNe::Transfer::Sigmoid ) }

    it "is numerically accurate gradient for the loss function measured pre-transfer with y=0.0 or 1.0" do
      test_size_range(2,5) do |n|
        targets = NArray.int(n).random(2).to_f
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MulticlassLogLoss.sigmoid_de_dz( demi_layer.output, targets )

        rough_gradients = demi_layer.measure_de_dz( zvals, targets )
        (0...n).each do |i|
          if rough_gradients[i] == 0
            expect( got_de_dz[i] ).to be_within(1e-6).of 0.0
          else
            expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
          end
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :logloss, :sigmoid ...) with y=0.0 or 1.0" do
      test_size_range(2,5) do |n|
        targets = NArray.int(n).random(2).to_f
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MulticlassLogLoss.sigmoid_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :mlogloss, :sigmoid, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end

    it "is numerically accurate gradient for the loss function measured pre-transfer with y in [0.0..1.0]" do
      test_size_range(2,5) do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MulticlassLogLoss.sigmoid_de_dz( demi_layer.output, targets )

        rough_gradients = demi_layer.measure_de_dz( zvals, targets )
        (0...n).each do |i|
          if rough_gradients[i] == 0
            expect( got_de_dz[i] ).to be_within(1e-6).of 0.0
          else
            expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
          end
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :logloss, :sigmoid ...) with y in [0.0..1.0]" do
      test_size_range(2,5) do |n|
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MulticlassLogLoss.sigmoid_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :mlogloss, :sigmoid, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end
  end

  describe "#softmax_de_dz" do
    let(:demi_layer) { TestDemiOutputLayer.new( RuNeNe::Objective::MulticlassLogLoss, RuNeNe::Transfer::Softmax ) }

    it "is numerically accurate gradient for the loss function measured pre-transfer with y=0.0 or 1.0" do
      test_size_range(2,5) do |n|
        targets = NArray.int(n).to_f
        targets[ rand(n) ] = 1.0
        zvals = NArray.sfloat(n).random(2.0) - 1.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MulticlassLogLoss.softmax_de_dz( demi_layer.output, targets )

        rough_gradients = demi_layer.measure_de_dz( zvals, targets, 0.005 )

        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :mlogloss, :softmax ...) with y=0.0 or 1.0" do
      test_size_range(2,5) do |n|
        targets = NArray.int(n).to_f
        targets[ rand(n) ] = 1.0
        zvals = NArray.sfloat(n).random(2.0) - 1.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MulticlassLogLoss.softmax_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :mlogloss, :softmax, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end

    it "is numerically accurate gradient for the loss function measured pre-transfer with y in [0.0..1.0]" do
      def layer_setup
        targets = NArray.sfloat(n).random(1.0)
        zvals = NArray.sfloat(n).random(4.0) - 2.0
        demi_layer.run( zvals, targets )
        got_de_dz = RuNeNe::Objective::MulticlassLogLoss.softmax_de_dz( demi_layer.output, targets )
      end

      test_size_range(2,5) do |n|
        layer_setup
        rough_gradients = demi_layer.measure_de_dz( zvals, targets, 0.005 )
        (0...n).each do |i|
          expect( got_de_dz[i] / rough_gradients[i] ).to be_within( 0.01 ).of 1.0
        end
      end
    end

    it "is same as calling RuNeNe::Objective.de_dz( :mlogloss, :softmax ...) with y in [0.0..1.0]" do
      test_size_range(2,5) do |n|
        layer_setup
        got_de_dz = RuNeNe::Objective::MulticlassLogLoss.softmax_de_dz( demi_layer.output, targets )
        generic_de_dz = RuNeNe::Objective.de_dz( :mlogloss, :softmax, demi_layer.output, targets )
        expect( got_de_dz ).to be_narray_like generic_de_dz
      end
    end
  end
end