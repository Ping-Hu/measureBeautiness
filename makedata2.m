%--------------------------------------------------------------------------
% This file is to cook data for the CNN model in nolearn.
% Training set:
% total: 20 * 75 * 44 images
% --->20 scores; in every score: 75 images, every original image is augmented into
% 44 images.
% 
%--------------------------------------------------------------------------


clear all;


% amount of images corresponding to one original image
num_per_oriImage = 4;
% amount of original images in one category
num_per_score = 75;
% image size
height = 100;
width = 75;

inputdir = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/FL/';
outputdir = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/nolearnTrain/';
if ~exist(outputdir,'dir')
    cmd = sprintf('mkdir %s', outputdir);
    system(cmd);
end

category = dir(inputdir);

while ~isempty(category(1).name)
    if (category(1).name(1) == '.')
        category(1) = [];
    else
        break;
    end
end

% amount of categories
caNumber = length(category);

%--------------------------------------------------------------------------
% for every i in the 'category'
%   get path of this category ---> path (inputdir + category(i))
%   get all the image folder in the category folder ---> in imageFolderList
%   remove the '.' folder
%   get the length of the imageFolderList
%   generate a K1(75) permutation as selected image folders' ID's ---> in selected
%
%   for every number j in the 'selected'
%       open up a new row of vector (100*75*44+1)
%       get the path of one selected image folder ---> selected_imageFolder
%           (path + imageFolderList(selected(j)))
%       get all the images from this path ---> image_candidates
%       remove the '.' images
%       get the length of the image_candidates
%       generate a K2(44) permutation as selected images' IDs ---> in
%           selected_indices
%       for every number in selected_indices
%           get the path of one selected image ---> in target_image_path 
%               (selected_imageFolder + image_candidates(selected_indices(s)))
%           access that image ---> in target_image
%           read, and store it as one part of the row vector corresponding
%               to the original image
%       end for
%
%    end for
%
% end for
%                    
%--------------------------------------------------------------------------

target_matrix = zeros(caNumber*num_per_score , 100*75*3*num_per_oriImage + 1);

for i = 1:caNumber
    fprintf('Start processing %s...\n', num2str(i)); 
    path = fullfile(inputdir,category(i).name);
    imageFolderList = dir(path);
    while ~isempty(imageFolderList(1).name)
        if (imageFolderList(1).name(1) == '.')
            imageFolderList(1) = [];
        else
            break;
        end
    end
    
    n = length(imageFolderList);
    % select 'num_per_score' image folders from 100 image folders
    selected = randperm(n, num_per_score);
    
    for j = 1: num_per_score
        fprintf('randomly pick image folder %s :\n', imageFolderList(selected(j)).name);
        selected_imageFolder = fullfile(...
            path,imageFolderList(selected(j)).name);
        
        image_candidates = dir(selected_imageFolder);
        while ~isempty(image_candidates(1).name)
            if (image_candidates(1).name(1) == '.')
                image_candidates(1) = [];
            else
                break;
            end
        end
        n = length(image_candidates);
        % pick 'num_per_oriImage' images from 440 images
        selected_indices = randperm(n, num_per_oriImage);
        target_row = zeros(1, 100*75*3*num_per_oriImage + 1);
        for s = 1:num_per_oriImage
            
            target_image_path = fullfile(...
                selected_imageFolder, image_candidates(selected_indices(s)).name);
            target_image = im2double(imread(target_image_path));
            % target_image is 100*75*3
            reshape_im = zeros(1,100*75*3);
            for t=1:3
                im = target_image(:,:,t)';
                im = reshape(im,[1,7500]);
                start_ind = (t-1)*100*75+1;
                end_ind = t*100*75;
                reshape_im(1,start_ind:end_ind) = im;
            end
            start_ind = (s-1)*100*75*3 + 1;
            end_ind = s*100*75*3;
            target_row(1,start_ind:end_ind) = reshape_im;
        end
        ind_x = (i-1)*num_per_score + j;
        if(100*75*3*num_per_oriImage+1 == 90001)
            ;%continue;
        end
        target_row(1,100*75*3*num_per_oriImage+1) = i;
        
        for h = 1:100*75*3*num_per_oriImage+1
%             fprintf('%s\n',num2str(target_row(1,h)));
%             fprintf('%s\n',num2str(target_matrix(ind_x,h)));
            target_matrix(ind_x,h) = target_row(1,h);
%             fprintf('%s\n',num2str(target_matrix(ind_x,h)));
        end
        
    end
    
%     fprintf('target_matrix %s-th row last number is %s',...
%         num2str((i)*num_per_score), num2str(target_matrix((i)*num_per_score,90001)));
%     
    fprintf('Category %s is completed.\n', num2str(i));
    
end

% outName = fullfile(outputdir, 'target_matrix.mat');
% save(outName, 'target_matrix', '-v7.3');
% 
% dataFileName = fullfile(outputdir,'target_matrix.mat');
% FileData = load(dataFileName);
csvFileName = fullfile(outputdir,'target_matrix.csv');
csvwrite(csvFileName, target_matrix);








