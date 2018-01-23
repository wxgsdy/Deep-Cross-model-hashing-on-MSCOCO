function [ word_embedding ] = word2code(word,vectors)
%word embedding
dictionary=vectors(:,1);
[~,index]=ismember(word,dictionary);

if index==0
   code=vectors(size(vectors,1),2:51); 
else
 code=vectors(index,2:51);  
end
word_embedding = cell2mat(code);

if index==0
  word_embedding(1,51) = 8913;
else
  word_embedding(1,51) = index;
end
end

