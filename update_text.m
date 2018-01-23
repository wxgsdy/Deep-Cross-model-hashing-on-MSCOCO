function [net2,state,vectors_mat,posarray]  = update_text(net2,learningRate,state,vectors_mat,posarray,forupdate,ix)

weightDecay = 5*1e-4 ;
batchSize = 100 ;
momentum = 0.9 ;

for p=1:numel(net2.params)
      thisDecay = weightDecay * net2.params(p).weightDecay ;
      thisLR = learningRate * net2.params(p).learningRate ;
      state.momentum_txt{p} = momentum * state.momentum_txt{p} ...
        - thisDecay * net2.params(p).value ...
        - (1 / (batchSize*5000)) * net2.params(p).der ;
      net2.params(p).value = net2.params(p).value + thisLR * state.momentum_txt{p};
%	fprintf('layer %d (%s) derivatives: max %e min %e\n',p,net2.params(p).name,max(abs(net2.params(p).der)),min(abs(net2.params(p).der)));

  end 
   
   % update word & position embedding  
 temp1 =  gpuArray(zeros(size(vectors_mat,1),50)); 
 vectors_num = gpuArray(vectors_mat);
 res = net2.vars(1).der;            %50*58*1*batchsize 
 for i=1:batchSize                  % every sen
     index = forupdate(:,:,ix(i));
     index(all(index==0,2),:)=[]; 	
	 len=size(index,1);              % every word
	 tt = res(:,:,:,i);              % 2-dimension
	 for j=1:len 
	   temp1(index(j,1),:) = temp1(index(j,1),:) + tt(j,1:50);
	 end
 end
 	
 state.momentum_word = momentum *  state.momentum_word...
        - weightDecay * vectors_num...
		- (1 / (batchSize*5000)) * temp1 ;
 
 % update word embedding 
   vectors_num = vectors_num + learningRate * state.momentum_word;
   vectors_mat = gather(vectors_num);
   
   
  temp2 =  gpuArray(zeros(50,4));
  res = net2.vars(1).der;
  posarray_gpu = gpuArray(posarray);
  for i=1:batchSize                  % every sen
     index = forupdate(:,:,ix(i));    
	 index(all(index==0,2),:)=[]; 	
	 len=size(index,1);               % every word
	 tt = res(:,:,:,i);               % 2-dimension
	 for j=1:len 
	   temp2(index(j,2),:) = temp2(index(j,2),:) + tt(j,51:54);
	   temp2(index(j,3),:) = temp2(index(j,3),:) + tt(j,55:58);
	 end
 end
 
 state.momentum_pos = momentum *  state.momentum_pos...
        - weightDecay * posarray_gpu - (1 / (batchSize*5000)) * temp2 ;
		
% update pos embedding		
posarray_gpu = posarray_gpu + learningRate * state.momentum_pos;
posarray = gather(posarray_gpu);