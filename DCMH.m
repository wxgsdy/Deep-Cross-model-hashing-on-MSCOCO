function [B_dataset,B_test,map] = DCMH()
    %% test
    clear;clc;
    codelens=32;     
    run matconvnet-beta23/matlab/vl_setupnn.m;  
    %% load DagNN network
    net=load('data/imagenet-vgg-f.mat');
     
	net1 = net_structure_image (net,codelens);
	net2 = net_structure_text(codelens);
	net1.move('gpu');
	net2.move('gpu');
	
	% prepare dataset
	load('data/vectors.mat');
	load('data/posarray.mat');
	vectors_mat = cell2mat(vectors(:,2:51));
	
	annFile='/media/chg/dataset/MSCOCO/captions_train2014.json';	
	coco=CocoApi(annFile);	
	[dataset_train_image,dataset_train_text,dataset_val_image,dataset_val_text,label_train,label_val,...
	    forupdate] = data_prepare (vectors,posarray)
    %load('dataset.mat');	
    
   % prepare F & G
      [F,G]= precalc(net1,net2,dataset_train_image,dataset_train_text);
	  B = sign(F+G);     
    
    
   % init parameters    
	
    imglr = 1e-2 ;
	textlr = 1e-2;
	  

    %% training	
    for iter = 1: maxIter
        [net1,net2,B,vectors_mat,posarray] = train(B,F,G,dataset_train_image,dataset_train_text,net1,...
                                 net2,imglr,textlr,label_train_mat,iter,vectors_mat,posarray,forupdate) ;
        %% learning rate changes
        if mod(iter,20)==0
            imglr = imglr*(2/3);
			textlr = textlr*(2/3);
        end
    end
    for iter = 1: maxIter
        [net1,net2,B,vectors_mat,posarray] = train(B,F,G,dataset_train_image,dataset_train_text,net1,...
                                 net2,imglr,textlr,label_train_mat,iter,vectors_mat,posarray,forupdate) ;
        %% learning rate changes
        if mod(iter,20)==0
            imglr = imglr*(2/3);
			textlr = textlr*(2/3);
        end
    end
    %diary off; 	
    %% testing
    [map,B_dataset,B_test] = test(net, dataset_L, test_L,data_set, test_data );
%end

%log_file=['./save files/log_',time_str,'.txt'];
    %diary(log_file);
	%diary off;  
	% time_str=datestr(now,30);%