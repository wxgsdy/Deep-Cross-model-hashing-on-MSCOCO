% output:
%    B_dataset
%	 B_testset

function [B_dataset, B_testset] = compute_B1 (net1,net2,dataset,testset)

% dataset 是 image, 且 testset为文本可直接计算	

	if ndims(dataset)==4
		im = dataset;
		im = imresize(im, [224,224]) ;
		im = im - repmat(net1.meta.normalization.averageImage,1,1,1,size(im,4)) ;
		
		for i=1:size(dataset,4)
			im_ = im(:,:,:,i);
			im_ = single(im_);
			im_ = gpuArray(im_);
			net1.eval({'data',im_});
			res=net1.vars(net1.getVarIndex('x20')).value ;
			F0 = squeeze(gather(res))';
			F(i,:)=F0;
		end
		B_dataset = sign(F);
		
		for i=1:size(testset,3)
			txt = dataset(:,:,i);
			txt = single(txt); 
			txt = gpuArray(txt);
			net2.eval({'input',txt});
			res = net2.vars(net2.getVarIndex('fc1')).value ;
			U0 = squeeze(gather(res))';
			G(i,:)=U0;
		end
		
		B_testset=sign(G);
		
		% txt = single(testset);	
		% txt=reshape(txt,[50,58,1,size(testset,3)]);
		% txt = gpuArray(txt) ;	  	
		% net2.eval({'input',txt}); 	
		% res = net2.vars(net2.getVarIndex('fc1')).value ;
		% G0 = squeeze(gather(res))';
		% B_testset = sign(G0);
		
% dataset是文本，都要分开计算  
    else 
		for i=1:size(dataset,3)
			txt = dataset(:,:,i);
			txt = single(txt); 
			txt = gpuArray(txt);
			net2.eval({'input',txt});
			res = net2.vars(net2.getVarIndex('fc1')).value ;
			U0 = squeeze(gather(res))';
			G(i,:)=U0;
		end
		B_dataset = sign(G);
				
		
		im = testset;
		im = imresize(im, net1.meta.normalization.imageSize(1:2)) ;
		im = im - repmat(net1.meta.normalization.averageImage,1,1,1,size(im,4)) ;
		for i=1:size(testset,4)
			im_ = im(:,:,:,i);
			im_ = single(im_);
			im_ = gpuArray(im_);
			net1.eval({'data',im_});
			res=net1.vars(net1.getVarIndex('x20')).value ;
			F0 = squeeze(gather(res))';
			F(i,:)=F0;
		end
		B_testset = sign(F);
end
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		