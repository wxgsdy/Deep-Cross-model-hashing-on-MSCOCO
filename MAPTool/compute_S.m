function S = compute_S (train_L,test_L)
    Dp=train_L*test_L';
    S=Dp>0;
end