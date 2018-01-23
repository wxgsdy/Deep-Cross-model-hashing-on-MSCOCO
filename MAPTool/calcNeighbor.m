% calculate a logic matrix indicating whether a pair of items are neighbors.
function R = calcNeighbor (label, idx1, idx2)
  
    L1 = label(idx1,:);
    L2 = label(idx2,:);
    L1 = single(L1);
    L2 = single(L2);
    Dp=L1*L2';
    R=Dp>0;

end
