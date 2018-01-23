% output:
%    B_dataset
%	 B_test

function [B_dataset, B_testset] = compute_B2 (net1,net2,dataset,testset)
  
	if ndims(dataset)==4
		image = dataset;
		text = testset;
	else
		image = testset;
		text = dataset;
	end 
  
%  net1.move('cpu');
%  net2.move('cpu');
	im = image;
	im = imresize(im, net1.meta.normalization.imageSize(1:2)) ;
	im = im - repmat(net1.meta.normalization.averageImage,1,1,1,size(im,4)) ;
  
	for i=1:size(image,4)
		im_ = im(:,:,:,i);
		im_ = single(im_);
		im_ = gpuArray(im_);
		net1.eval({'data',im_});
		res=net1.vars(net1.getVarIndex('x20')).value ;
		F0 = squeeze(gather(res))';
		F(i,:)=F0;
	end
	B_image = sign(F);
	
	
	txt = single(text);	
	txt=reshape(txt,[50,58,1,size(text,3)]);
	txt = gpuArray(txt) ;	  	
	net2.eval({'input',txt}); 	
	res = net2.vars(net2.getVarIndex('fc1')).value ;
	G0 = squeeze(gather(res))';
	B_text = sign(G0);
	
	if ndims(dataset)==4
		B_dataset = B_image;
		B_testset = B_text;
	else
		B_dataset = B_text;
		B_testset = B_image;
	end 
 
	% for j=1:size(text,3)
		% txt = text(:,:,j);
		% txt(all(txt==0,2),:)=[];
		% txt=single(txt); 
		% net2.eval({'input',txt});
		% res=net2.vars(net2.getVarIndex('fc1')).value ;
		% U0 = squeeze(res);
		% G(j,:)=U0;
	% end
 
  
	 
%	 net1.move('gpu');
%    net2.move('gpu');
end
 