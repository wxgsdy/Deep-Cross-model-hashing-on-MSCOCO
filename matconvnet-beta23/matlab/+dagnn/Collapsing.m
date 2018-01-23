classdef Collapsing < dagnn.ElementWise
% Collapsing calculates the sums of every row of the input matrix 

properties(Transient)
 numInputs
end

methods

    function outputs = forward(obj, inputs, params)
	  obj.numInputs = numel(inputs) ;
	 
      outputs{1} = sum(inputs{1},2);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)     
	 
      temp = gpuArray(zeros(size(inputs{1},1),size(inputs{1},2),size(inputs{1},3),size(inputs{1},4)));
	  for i= 1:size(inputs{1},2)	
	  	temp(:,i,:,:) = derOutputs{1};
	  end
	  temp = single(temp);
      derInputs{1} = temp;
      derParams = {};    
    end 

%    function outputSizes = getOutputSizes(obj, inputSizes)
%       outputSizes{1} = [inputSizes{1}(1) 1 inputSizes{1}(3)] ;
      
%    end
 
    function obj = Collapsing(varargin)
      obj.load(varargin) ;
    end
  end
end