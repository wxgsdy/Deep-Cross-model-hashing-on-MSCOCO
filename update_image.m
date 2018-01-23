function [net1,state]  = update_image(net1,learningRate,state)

weightDecay = 5*1e-4 ;
batchSize = 100 ;
momentum = 0.9 ;  


  for p=1:numel(net1.params)
      thisDecay = weightDecay * net1.params(p).weightDecay ;
      thisLR = learningRate * net1.params(p).learningRate ;
      state.momentum_im{p} = momentum * state.momentum_im{p} ...
        - thisDecay * net1.params(p).value ...
        - (1 / (batchSize*5000)) * net1.params(p).der ;
      net1.params(p).value = net1.params(p).value + thisLR * state.momentum_im{p};

   end 
end
