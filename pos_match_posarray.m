function [ representation ] = pos_match_posarray( pos,posarray )
% look up the posarray matrix to find position representation
for j=1:size(pos,1)
     for i=1:size(posarray,1)
      if pos{j,2}==posarray(i,1) && pos{j,3}==posarray(i,2)
        representation(j,51)=posarray(i,3);
        break;
      end
     end
   end


end

