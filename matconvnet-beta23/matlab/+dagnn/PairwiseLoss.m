classdef PairwiseLoss < dagnn.ElementWise
  properties (Transient)
    U
    S
    ix
  end

  methods
    function outputs = forward(obj, inputs, params)
        
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        U0 = squeeze(inputs{1})';
        obj.U(obj.ix,:) = U0 ;
        T = U0 * obj.U' / 2 ;
        A = 1 ./ (1 + exp(-T)) ;
        derInputs{1}=(A-obj.S) * obj.U ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
