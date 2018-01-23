%input:
%        dataset_image;
%        test_text;
%        dataset_image_L;
%		 test_text_L;


function [map,B_dataset,B_test] = test(net1,net2,dataset,testset,label_dataset,label_test)
    S = compute_S(label_dataset,label_test) ;
    [B_dataset, B_test] = compute_B2 (net1,net2,dataset,testset);
    map = return_map (B_dataset, B_test, S) ;
end