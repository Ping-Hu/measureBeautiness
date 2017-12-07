function imageflipping( infilename, outputPath )
% this function crops an image according to the window's size


if(nargin < 2)
    fprintf('Lacking parameters. Processing is stopped! /n');
    exit();
end

[path,name,ext] = fileparts(infilename);

image = im2double(imread(infilename));
[nr,nc,nd] = size(image);
flippedI = zeros(nr,nc,nd);


for i = 1:nr
    for j = 1:nc
        % row's index will not be changed! only flip the left-and-right
        % side!
        newi = i;
        newj = nc-j+1;
        flippedI(i,j,1:nd) = image(newi, newj, 1:nd);
        
    end
end

%prefix = ['disflip' '-' ];
resultname = fullfile(outputPath, [name ext]);
imwrite(image, resultname);

prefix = ['flip' '-' ];
resultname = fullfile(outputPath, [prefix name ext]);
imwrite(flippedI, resultname);

end

