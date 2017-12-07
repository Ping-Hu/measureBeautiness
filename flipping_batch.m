function flipping_batch( inPath, outPath )
% this function produces two images, one is the input image, the other one
% is the flipped image.

if ~exist(outPath,'dir')
    cmd = sprintf('mkdir %s', outPath);
    system(cmd);
end
category = dir(inPath);
for i = 1: length(category)
    if category(i).name(1) == '.'
        continue
    elseif strcmp(category(i).name(1),'..')
        continue
    elseif strcmp(category(i).name(1),'.DS_Store')
        continue
    else
        
        fprintf('processing %s\n', category(i).name);
        deeper_inPath = fullfile(inPath, category(i).name);
        imageFolder = dir(deeper_inPath);
        for j = 1:length(imageFolder)
            if imageFolder(j).name(1) == '.'
                continue;
            else
                real_inPath = fullfile(deeper_inPath, imageFolder(j).name);
                iname = fullfile(real_inPath,'*');
                imageFile = dir(iname);
                outputPath = fullfile(outPath,category(i).name);
                if ~exist(outputPath,'dir')
                    cmd = sprintf('mkdir %s', outputPath);
                    system(cmd);
                end
                outputPath = fullfile(outputPath, imageFolder(j).name);
                if ~exist(outputPath,'dir')
                    cmd = sprintf('mkdir %s', outputPath);
                    system(cmd);
                end
                for k = 1: length(imageFile)
                    if imageFile(k).name(1) == '.'
                        continue
                    elseif strcmp(imageFile(k).name(1),'..')
                        continue
                    else
                        filename = fullfile(real_inPath, imageFile(k).name);
                        imageflipping(filename, outputPath);
                    end
                end
            end
        end
    end
end

