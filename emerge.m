function [ representation ] = emerge( sen,vectors,posarray )
% output the whole embedding for the sentence
% vectors is the pre-trained word embedding table,posarray is the pre-initialized embedding table

pos=getpos(sen);
%load('data/posarray.mat');
length=size(pos,1);
representation=[];

%word representation
for i=1:length    
    representation(i,1:51)= word2code(pos{i,1},vectors);;
end

%position representation
for i=1:size(pos,1)
   representation(i,52:55)=posarray(pos{i,2},1:4);
   representation(i,56:59)=posarray(pos{i,3},1:4);
   representation(i,60) = pos{i,2};
   representation(i,61) = pos{i,3};  
end


end

