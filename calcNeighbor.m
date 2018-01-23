% calc the similarity matrix between the selected image & text batches
function R = calcNeighbor (label, idx1, idx2)

  N1 = length(idx1);
  N2 = length(idx2);
  L1 = label(idx1);
  L2 = label(idx2);

  for i=1:N1
   for j=1:N2
       a=single(L1{i});
       b=single(L2{j});       
       R(i,j)= ~isempty(intersect(a,b));
   end 
  end
 end

