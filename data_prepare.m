% prepare dataset for training and validation and pre_train B;
% target output:dataset_train_image; 
%               dataset_train_text;
%				dataset_val_image;
%				dataset_val_test;
%               label_train
%               label_val
%               label_train_mat;
%               label_val_mat;


function [dataset_train_image,dataset_train_text,dataset_val_image,dataset_val_text,label_train,label_val,forupdate] = data_prepare (vectors,posarray)

dataset_train_text=zeros(50, 58, 5000);
dataset_val_text=zeros(50, 58, 1000);
forupdate = zeros(50,3,5000);
label_train_mat = zeros(5000,90);
label_val_mat = zeros(1000,90);


%prepare training & validation set
  annFile = '/media/chg/dataset/MSCOCO/captions_train2014.json';	
  coco = CocoApi(annFile);  
  annFile = '/media/chg/dataset/MSCOCO/instances_train2014.json';	
  coco_instances = CocoApi(annFile); 
  index = randi(80000,6000,1);


  acci=80001;
% net=load('imagenet-vgg-f.mat')
  
% training set  
  for i = 1:5000	  
      img = coco.loadImgs(coco.data.images(index(i)).id);	  
      I= imread(sprintf('/media/chg/dataset/MSCOCO/train2014/%s',img.file_name));
	  id=coco.data.images(index(i)).id;
      while ndims(I) ~= 3          
		  img = coco.loadImgs(coco.data.images(acci).id);
          I= imread(sprintf('/media/chg/dataset/MSCOCO/train2014/%s',img.file_name));
		  id=coco.data.images(acci).id;
		  acci = acci + 1;		  
	  end	  
	  I = imresize(I,[224 224]) ;
	  dataset_train_image(:,:,:,i) = single(I);
	  
	  annIds = coco.getAnnIds('imgIds',id);
	  anns=coco.loadAnns(annIds);
	  sen = anns(1).caption;      
      sen=strtrim(sen);
      sen=lower(sen);
      punctuation='[\[\]\.({!),:;?}"]';
      sen=regexprep(sen,punctuation,'');
	  rep=emerge( sen,vectors,posarray );	         % 61 bits
      len=size(rep,1);
	  dataset_train_text(1:len,1:50,i) = single(rep(1:len,1:50));
	  dataset_train_text(1:len,51:58,i) = single(rep(1:len,52:59));
	  forupdate(1:len,1,i) = rep(1:len,51);
	  forupdate(1:len,2,i) = rep(1:len,60);
	  forupdate(1:len,3,i) = rep(1:len,61);
	  
	  
	  annIds=coco_instances.getAnnIds('imgIds',id);
	  anns=coco_instances.loadAnns(annIds);
	  for t=1:size(anns,2)
        cator(t)=anns(t).category_id;
      end
	  cator=unique(cator);
      label_train{i}=cator;
	end
 
 % validation set
 for i = 5001:6000
      img = coco.loadImgs(coco.data.images(index(i)).id);	  
      I= imread(sprintf('/media/chg/dataset/MSCOCO/train2014/%s',img.file_name));
	  id=coco.data.images(index(i)).id;
      while ndims(I) ~= 3          
		  img = coco.loadImgs(coco.data.images(acci).id);
          I= imread(sprintf('/media/chg/dataset/MSCOCO/train2014/%s',img.file_name));
		  id=coco.data.images(acci).id;
		  acci = acci + 1;		  
	  end	  
	  I = imresize(I,[224 224]) ;
	  dataset_val_image(:,:,:,i-5000) = single(I);
	  
	  annIds = coco.getAnnIds('imgIds',id);
	  anns=coco.loadAnns(annIds);
	  sen = anns(1).caption;      
      sen=strtrim(sen);
      sen=lower(sen);
      punctuation='[\[\]\.({!),:;?}"]';
      sen=regexprep(sen,punctuation,'');
	  rep=emerge( sen,vectors,posarray );	  
      len=size(rep,1);
	  dataset_val_text(1:len,1:50,i-5000) = single(rep(1:len,1:50));
	  dataset_val_text(1:len,51:58,i-5000) = single(rep(1:len,52:59));
	  
	  annIds=coco_instances.getAnnIds('imgIds',id);
	  anns=coco_instances.loadAnns(annIds);
	  for t=1:size(anns,2)
        cator(t)=anns(t).category_id;
      end
	  cator=unique(cator);
      label_val{i-5000}=cator;
	end

% convert label martrix to mat	
	for i=1:5000
     tt = label_train{i};
     for j=1:length(tt)
        label_train_mat(i,tt(j))=1;
     end
	end
	
	for i=1:1000
     tt = label_val{i};
     for j=1:length(tt)
        label_val_mat(i,tt(j))=1;
     end
	end

	
	
	
	
	
end
	
    
	 