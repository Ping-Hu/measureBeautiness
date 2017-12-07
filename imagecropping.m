function imagecropping( infilename, outputPath, winR, winC, wstrideR, wstrideC )
% this function crops an image according to the window's size

if(nargin < 2)
    fprintf('Lacking parameters. Processing is stopped! /n');
    exit();
end
if(wstrideR == 0)||(wstrideC == 0)
    fprintf('stride cannot be zero. Processing is stopped! /n');
    exit();
end


[path,name,ext] = fileparts(infilename);

image = imread(infilename);
[nr,nc,nd] = size(image);
rcount = (nr-winR)/wstrideR+1;
ccount = (nc-winC)/wstrideC+1;
for i = 1:rcount
    for j = 1:ccount
        rstart = 1+wstrideR*(i-1);
        rend   = winR+wstrideR*(i-1);
        cstart = 1+wstrideC*(j-1);
        cend   = winC+wstrideC*(j-1);
        
        croppedI = image(rstart:rend,cstart:cend,1:nd);
        
        prefix = ['crop' int2str(i) '-' int2str(j)  '-' ];
        
        resultname = fullfile(outputPath, [prefix name ext]);
        imwrite(croppedI, resultname);
        
    end
end



end

