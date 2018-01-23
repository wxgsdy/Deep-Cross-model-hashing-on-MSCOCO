% text query vs images database 
% prepare testset & dataset for testset : 5 sens,100 images out of each class
% target output:
%        dataset_image;   224*224*3*8000
%        dataset_text;    50*58*8000
%        test_text;       50*58*400
%        test_image;      224*224*3*400
%        label_dataset;   8000*90
%		 label_test;      400*90 
%        label_dataset_cell;
%		 label_test_cell;

function [dataset_image, dataset_text, test_text, test_image, label_dataset, label_test] = datafortest (vectors,posarray)
 		
  annFile = '/media/chg/dataset/MSCOCO/captions_train2014.json';	
  coco = CocoApi(annFile);  
  annFile = '/media/chg/dataset/MSCOCO/instances_train2014.json';	
  coco_instances = CocoApi(annFile);  
 
  test_text = zeros(50,58,400);
  dataset_text = zeros(50,58,8000);  
  label_dataset = zeros(8000,90);
  label_test = zeros(400,90);  
  label_dataset_cell={};
  label_test_cell={};
  
 for i = 1:80
    imgIds_total = coco_instances.getImgIds('catIds',coco_instances.inds.catIds(i));
	num = size(imgIds_total,1);
	index = randperm(num) ;
	imgIds = imgIds_total(index(1:105));
	acci = 106;
	
% test
	for j = 1:5
		img = coco.loadImgs(imgIds(j));	  
		I= imread(sprintf('/media/chg/dataset/MSCOCO/train2014/%s',img.file_name));
		id=imgIds(j);	 
	    while ndims(I) ~= 3          
			img = coco.loadImgs(imgIds_total(index(acci)));
			I= imread(sprintf('/media/chg/dataset/MSCOCO/train2014/%s',img.file_name));
			id=imgIds_total(index(acci));			
			acci = acci + 1;		  
		end	
		
	% test_image
	    I = single(I);
		I = imresize(I,[224 224]) ;
		test_image(:,:,:,j+(i-1)*5) = I;
	% test_text	
		annIds = coco.getAnnIds('imgIds',id);
		anns=coco.loadAnns(annIds);
		sen = anns(1).caption;
		sen=lower(sen);
		punctuation='[\[\]\.({!),:;?}"]';
		sen=regexprep(sen,punctuation,'');
		rep=emerge( sen,vectors,posarray );	         % 61 bits
		len=size(rep,1);	  	
		test_text(1:len,1:50,j+(i-1)*5) = single(rep(1:len,1:50));
		test_text(1:len,51:58,j+(i-1)*5) = single(rep(1:len,52:59));
	  
    % label_test	  
	  annIds=coco_instances.getAnnIds('imgIds',id);
	  anns=coco_instances.loadAnns(annIds);
	  for t=1:size(anns,2)
        cator(t)=anns(t).category_id;
      end
	  cator=unique(cator);
	  label_test_cell{j+(i-1)*5}=cator;	

 %   fprintf('processing test：%d\n',j+(i-1)*5);  
	end

	
% dataset	
	for j=6:105
	
	% dataset_image
		img = coco.loadImgs(imgIds(j));	  
		I= imread(sprintf('/media/chg/dataset/MSCOCO/train2014/%s',img.file_name));
		id=imgIds(j);	  
		while ndims(I) ~= 3          
			img = coco.loadImgs(imgIds_total(index(acci)));
			I= imread(sprintf('/media/chg/dataset/MSCOCO/train2014/%s',img.file_name));
			id=imgIds_total(index(acci));
			acci = acci + 1;		  
		end	
		I = single(I);
		I = imresize(I,[224 224]) ;
		dataset_image(:,:,:,j-5+(i-1)*100) = single(I);
	
	% dataset_text	
		annIds = coco.getAnnIds('imgIds',id);
		anns=coco.loadAnns(annIds);
		sen = anns(1).caption;
		sen=lower(sen);
		punctuation='[\[\]\.({!),:;?}"]';
		sen=regexprep(sen,punctuation,'');
		rep=emerge( sen,vectors,posarray );	         % 61 bits
		len=size(rep,1);	  	
		dataset_text(1:len,1:50,j-5+(i-1)*100) = single(rep(1:len,1:50));
		dataset_text(1:len,51:58,j-5+(i-1)*100) = single(rep(1:len,52:59));
		
	  
	% dataset_image_L
		annIds=coco_instances.getAnnIds('imgIds',id);
		anns=coco_instances.loadAnns(annIds);
		for t=1:size(anns,2)
			cator(t)=anns(t).category_id;
		end
		cator=unique(cator);
		label_dataset_cell{j-5+(i-1)*100}=cator;
		
%	fprintf('processing dataset：%d\n',j-5+(i-1)*100); 	
	end
end	 

  
% convert label martrix to mat	
	for i=1:400
     tt = label_test_cell{i};
     for j=1:length(tt)
        label_test(i,tt(j))=1;
     end
	end
	
	for i=1:8000
     tt = label_dataset_cell{i};
     for j=1:length(tt)
        label_dataset(i,tt(j))=1;
     end
	end
end

