function contrast_batch( inPath, outPath, stepChange, bright )
% this function produces images with adjusted contrast and brightness
% inPath = 'absolute path..../dataset_face/raw2000/faces/'; its subfolders
% are 1, 2, 3,...
% outPath = 'absolute path..../dataset_face/preprocess/CB/'; its subfolders
% are 1, 2, 3,...

kLowContrastSteps = 5;
kHighContrastSteps = 5;
kChangePerStep = stepChange; %e.g. 0.1 will have contrast steps 0.4, 0.5, 0.6, 0.7...
%have user select files
%[files,pth] = uigetfile({'*.bmp;*.jpg;*.png;*.tiff;';'*.*'},'Select the Image[s]', 'MultiSelect', 'on');

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
        
        iname = fullfile(inPath, category(i).name,'*');
        imageFile = dir(iname);
        oname = fullfile(outPath,category(i).name);
        if ~exist(oname,'dir')
            cmd = sprintf('mkdir %s', oname);
            system(cmd);
        end
        for j = 1: length(imageFile)
            for s = -kLowContrastSteps : kHighContrastSteps
                step = 0.5 + (kChangePerStep * s);
                label = ['g', int2str(round( step*100) )];
                if imageFile(j).name(1) == '.'
                    continue
                elseif strcmp(imageFile(j).name(1),'..')
                    continue
                else
                    nam = fullfile(inPath,category(i).name, imageFile(j).name);
                    % create a folder for an original image
                    [pa, na, ex] = fileparts(imageFile(j).name);
                    o_name = fullfile(oname, na);
                    if ~exist(o_name,'dir')
                        cmd = sprintf('mkdir %s', o_name);
                        system(cmd);
                    end
                    if step < 0.5 %if reducing contrast, use linear transform
                        bmp_contrast(nam,o_name,step,bright,true,label,false);
                    else %if increasing contrast, use non-linear transform
                        bmp_contrast(nam,o_name,step,bright,false,label, false);
                    end;
                end
            end
        end
    end
end


