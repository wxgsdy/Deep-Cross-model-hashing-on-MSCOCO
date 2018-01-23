function [net1,net2,state,vectors,posarray] = update (net1, net2, learningRate, mode,... 
       ix, forupdate, state,vectors,posarray)

weightDecay = 5*1e-4 ;
batchSize = 100 ;
momentum = 0.9 ;
  
if mode == 'image'
  for p=1:numel(net1.params)
      thisDecay = weightDecay * net1.params(p).weightDecay ;
      thisLR = learningRate * net1.params(p).learningRate ;
      state.momentum_im{p} = momentum * state.momentum_im{p} ...
        - thisDecay * net1.params(p).value ...
        - (1 / batchSize) * net1.params(p).der ;
      net1.params(p).value = net1.params(p).value + thisLR * state.momentum_im{p};

   end 
 end  

if mode=='text'
  
  for p=1:numel(net2.params) 
      thisDecay = weightDecay * net2.params(p).weightDecay ;
      thisLR = learningRate * net2.params(p).learningRate ;
      state_momentum_txt{p} = momentum * state.momentum_txt{p} ...
        - thisDecay * net2.params(p).value ...
        - (1 / batchSize) * net2.params(p).der ;
      net2.params(p).value = net2.params(p).value + thisLR * state.momentum_txt{p} ;
   end 
   
% update word & position embedding  
 temp1 =  gpuArray(zeros(size(vectors,1),50)); 
 vectors_num = gpuArray(cell2mat(vectors(:,2:51)));
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
		- (1 / batchSize) * temp1 ;
 
 % update word embedding 
   vectors_num = vectors_num + learningRate * state.momentum_word;
   vectors(:,2:51)=num2cell(gather(vectors_num));
   
  temp2 =  gpuArray(zeros(50,4));
  res = net2.vars(1).der;
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
        - weightDecay * posarray - (1 / batchSize) * temp2 ;
		
% update pos embedding		
posarray = posarray + learningRate * state.momentum_pos;

end