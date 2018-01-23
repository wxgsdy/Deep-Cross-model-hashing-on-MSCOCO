classdef K_max_pooling < dagnn.ElementWise
  %   DagNN K_max_pooling layer
  %   The SUM layer takes the k maximun values from the input martrix(k=2)

  properties (Transient)    
	
	
  end

  methods
    function outputs = forward(obj, inputs, params)
      [resort,~]=sort(inputs{1},1);
	  outputs{1}=resort(end-1:end,:,:,:);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      
	  [~,index]=sort(inputs{1},1);
	  temp = gpuArray(zeros(size(inputs{1},1),size(inputs{1},2),size(inputs{1},3),size(inputs{1},4)));
	  %tt=gather(derOutputs{1});
	  
	  for i=1:size(inputs{1},4)
	    for j=1:size(inputs{1},3)
	      temp(index(end,:,j,i),:,j,i) = derOutputs{1}(2,:,j,i);
	      temp(index(end-1,:,j,i),:,j,i) = derOutputs{1}(1,:,j,i);
		end
	  end
	  
	  derInputs{1} = temp;	  
	  derParams = {} ;
    end

    function obj = K_max_pooling(varargin)
      obj.load(varargin) ;
    end
  end
end
