function [F,G] = precalc(net1,net2,dataset_train_image,dataset_train_text)
% run the initialized net with dataset

  im = dataset_train_image;
  im = imresize(im, net1.meta.normalization.imageSize(1:2)) ;
  im = im - repmat(net1.meta.normalization.averageImage,1,1,1,size(im,4)) ;

for i=1:size(dataset_train_image,4)
 im_ = im(:,:,:,i);
 im_ = single(im_);
 net1.eval({'data',im_});
 res=net1.vars(net1.getVarIndex('x20')).value ;
 U0 = squeeze(res)';
 F(i,:)=U0;
 
 fprintf('processing:%d\n',i);
 end
 
 for j=1:size(dataset_train_text,3)
 txt = dataset_train_text(:,:,j);
 txt(all(txt==0,2),:)=[];
 txt=single(txt); 
 net2.eval({'input',txt});
 res=net2.vars(net2.getVarIndex('fc1')).value ;
 U0 = squeeze(res)';
 G(j,:)=U0;
 fprintf('processing:%d\n',j);
 end
 
 
end
 
 