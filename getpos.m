function [ pos ] = getpos(sentence_in )
% get the position of each word in the sentence

S = regexp(sentence_in, ' ', 'split');
len=size(S,2);
pos=cell(len,3);
for i=1:len
  pos(i,1)=S(1,i);
  pos{i,2}=i;
  pos{i,3}=len+1-i;
end


end

