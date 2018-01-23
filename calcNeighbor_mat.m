% calc the similarity matrix between the selected image & text batches
function R = calcNeighbor_mat (label, idx1, idx2)

  % N1 = length(idx1);
  % N2 = length(idx2);
  % L1 = label(idx1,:);
  % L2 = label(idx2,:);
  
  % for i=1:N1
   % for j=1:N2  
     % a = L1(i,:);
     % b = L2(j,:); 
	 % L=a-b;
	 % t1=sum(a);        
	 % t2=length(find(L==1));  % nums of 1
	 % R(i,j)=t1>t2;
   % end
  % end
  
  a = label(idx1,:);
  b = label(idx2,:);
  Dp= a*b';
  R=Dp>0;
  
end 
	 