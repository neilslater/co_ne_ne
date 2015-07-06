require 'helpers'

describe RuNeNe::Objective::MeanSquaredError do
  describe "#loss" do
    it "is 0.0 when predictions and targets match" do
      (1..5).each do |n|
        5.times do
          targets = NArray.sfloat(n).random(20.0) - 10.0
          expect( RuNeNe::Objective::MeanSquaredError.loss( targets, targets ) ).to be_within(1.0e-10).of 0.0
        end
      end
    end

    it "is larger when predictions and targets are further apart" do
      targets = NArray.cast( [ 1.0, 2.0, 3.0, 4.0 ], 'sfloat' )
      preds1 =  NArray.cast( [ 1.5, 1.5, 3.5, 3.5 ], 'sfloat' )
      preds2 =  NArray.cast( [ 1.25, 1.0, 3.25, 5.0 ], 'sfloat' )

      expect( RuNeNe::Objective::MeanSquaredError.loss( preds1, targets ) ).to be < RuNeNe::Objective::MeanSquaredError.loss( preds2, targets )
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
      (1..5).each do |n|
        5.times do
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
  end
end


describe RuNeNe::Objective::LogLoss do
  describe "#loss" do
    it "is 0.0 when predictions and targets match" do
      (1..5).each do |n|
        5.times do
          targets = NArray.int(n).random(2).to_f
          expect( RuNeNe::Objective::LogLoss.loss( targets, targets ) ).to be_within(1.0e-10).of 0.0
        end
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
    it "is numerically accurate gradient for the loss function" do
      (1..5).each do |n|
        5.times do
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
    end
  end
end