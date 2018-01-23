classdef Conv_DC < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
  end
  
  properties (Transient)
    batchSize
    mask
  end

  methods
    function outputs = forward(obj, inputs, params)
    if strcmp(obj.net.mode, 'test')
      outputs{1} = vl_nnconv(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
        return ;
    end
      if ~obj.hasBias, params{2} = [] ; end
      % params{3} is mask matrix {[1 1 4096 codelens]*batchsize}     
    for k=1:obj.batchSize
      outputs{1}(:,:,:,k) = vl_nnconv(...
        inputs{1}(:,:,:,k), params{1}.*obj.mask(:,:,1:end-1,:,k), params{2}.*reshape(obj.mask(:,:,end,:,k),size(params{2})), ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
    end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if strcmp(obj.net.mode, 'test')
        [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
        return ;
      end
        
      if ~obj.hasBias, params{2} = [] ; end
      if isa(params,'gpuArray')
          derParams{1}=gpuArray(cast(zeros(obj.size),'like',params{1}));
      else
          derParams{1}=cast(zeros(obj.size),'like',params{1});
      end
      for k=1:obj.batchSize
      [derInputs{1}(:,:,:,k), derParams1_tmp, derParams{2}] = vl_nnconv(...
        inputs{1}(:,:,:,k), params{1}.*obj.mask(:,:,1:end-1,:,k), params{2}.*reshape(obj.mask(:,:,end,:,k),size(params{2})), derOutputs{1}(:,:,:,k), ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        obj.opts{:}) ;
        derParams{1}=derParams{1}+derParams1_tmp;
      end    
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      % he_gain: sc=0.01*sqrt(2 / prod(obj.size(1:3))) 
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') * sc ;
      end    
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = Conv_DC(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
