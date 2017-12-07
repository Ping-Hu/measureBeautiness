function  softwindow_batch(inPath, outPath, winR, winC, wstrideR, wstrideC)
% this function processes a batch of cropped images existed in the input path
%
% inPath = 'absolute path..../dataset_face/preprocess/CB/'; its subfolders
% are 1, 2, 3,...
% outPath = 'absolute path..../dataset_face/preprocess/SW/'; its subfolders
% are 1, 2, 3,...

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
        for j=1:length(imageFolder)
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
                        imagecropping(filename, outputPath, winR, winC, wstrideR, wstrideC);
                    end
                end
            end
        end
    end
end











