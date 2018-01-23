clear
resFile='../../captions_train2014.json';
res=gason(fileread(resFile));
fid= fopen('text.txt','wt');

for i=1:414113;
    temp=res.annotations(i).caption;
    temp=strtrim(temp);
    temp=lower(temp);
    punctuation='[\[\]\.({!),:;?}"]';
    temp=regexprep(temp,punctuation,'');
    
    fprintf(fid,'%c',temp);
    fprintf(fid,'\r\n');
    
end 

fclose(fid);




    