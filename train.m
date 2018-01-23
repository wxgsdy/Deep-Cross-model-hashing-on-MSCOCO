function [net1,net2,B,vectors_mat,posarray] = train(B,F,G,dataset_train_image,dataset_train_text,net1,...
                                 net2,imglr,textlr,label_train_mat,iter,vectors_mat,posarray,forupdate) 
								 
% training algorithm for net1 & net2

    N = size(dataset_train_image,4) ;
    batchsize = 100 ;
    index = randperm(N) ;
	state.momentum_im = num2cell(zeros(1, numel(net1.params))) ;  
    state.momentum_txt = num2cell(zeros(1, numel(net2.params))) ;  
    state.momentum_word = 0;
    state.momentum_pos = 0;

% train net1 with dataset_train_image	
	for j = 0:ceil(N/batchsize)-1
     	batch_time=tic ;		
		%% random select a minibatch
        ix = index((1+j*batchsize):min((j+1)*batchsize,N));
		S = calcNeighbor_mat (label_train_mat, ix, 1:N) ;
		im = dataset_train_image(:,:,:,ix) ;
        im_ = single(im) ; % note: 0-255 range
        im_ = im_ - repmat(net1.meta.normalization.averageImage,1,1,1,size(im_,4)) ;
	    im_ = gpuArray(im_) ;
		
		net1.eval({'data',im_}); 
		res = net1.vars(net1.getVarIndex('x20')).value ;
		F0 = squeeze(gather(res))';
		F(ix,:) = F0;		
		T = F0 * G' / 2 ;
		A = 1 ./ (1 + exp(-T)) ;
		dJdF = (A - S)*G / 2 + 2 *(F0-B(ix,:));
		
		dJdoutput = gpuArray(reshape(dJdF',[1,1,size(dJdF',1),size(dJdF',2)]));
		net1.eval({'data',im_},{'x20',dJdoutput}) ;
		
		[net1,state]  = update_image(net1,imglr,state);

		
		batch_time = toc(batch_time) ;		
	 fprintf(' image_iter %d  batch %d/%d (%.1f images/s) ,lr is %.3d \n', iter, j+1,ceil(N/batchsize), batchsize/ batch_time,imglr) ;
   end	 
% train net2 with dataset_train_text	

for j = 0:ceil(N/batchsize)-1       
		batch_time=tic ;	
		%% random select a minibatch
        ix = index((1+j*batchsize):min((j+1)*batchsize,N));
		S = calcNeighbor_mat (label_train_mat, ix, 1:N);
		txt = dataset_train_text(:,:,ix) ;
        txt=reshape(txt,[50,58,1,batchsize]);
        txt_ = single(txt) ;     
	    txt_ = gpuArray(txt_) ;	  	
		net2.eval({'input',txt_}); 
		res = net2.vars(net2.getVarIndex('fc1')).value ;
		G0 = squeeze(gather(res))';
		
		G(ix,:) = G0;
		T = G0 * F' / 2 ;
		A = 1 ./ (1 + exp(-T)) ;
		dJdG = (A - S)*F / 2 + 2 *(G0-B(ix,:));
		fprintf('\n');
		
	    fprintf('hash gradient mean: %e max: %e min: %e\n',mean2(dJdG),max(max(dJdG)),min(min(dJdG)));
		
		dJdoutput = gpuArray(reshape(dJdG',[1,1,size(dJdG',1),size(dJdG',2)]));
		net2.eval({'input',txt_},{'fc1',dJdoutput}) ;
		
		[net2,state,vectors_mat,posarray]  = update_text(net2,textlr,state,vectors_mat,posarray,forupdate,ix);
	
	 batch_time = toc(batch_time) ;
	 fprintf('***hash output mean:%e  max: %e min: %e**\n ',mean2(G0),max(max(G0)),min(min(G0)));
	 fprintf(' text_iter %d  batch %d/%d (%.1f sentences/s) ,lr is %.3d \n', iter, j+1,ceil(N/batchsize), batchsize/ batch_time,textlr) ;
	 
	end
	
 % calculate B
   B = sign(F+G);
end