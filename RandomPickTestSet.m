% This script is for partition the data set into test set and training set
% (trainning set includes trainning and validation)


path = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/raw2000/faces';
outputpath_test = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/nolearn/test';
outputpath_train = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/nolearn/train';
folders = dir(path);
while ~isempty(folders(1).name)
    if (folders(1).name(1) == '.')
        folders(1) = [];
    else
        break;
    end
end

fprintf('Get all %s folders.', num2str(length(folders)));


for i = 1:length(folders)
    deep_path = fullfile(path, folders(i).name);
    files = dir(deep_path);
    while ~isempty(files(1).name)
        if (files(1).name(1) == '.')
            files(1) = [];
        else
            break;
        end
    end
    
    fprintf('Get all %s files in score %s.\n', num2str(length(files)), num2str(folders(i).name));
    
    select = randperm(100, 100);
    for j = 1:25
        
        in_name = fullfile(path, folders(i).name, files(select(j)).name);
        im = imread(in_name);
        im = im2double(im);
        
        ou_path = fullfile(outputpath_test, folders(i).name);
        if ~exist(ou_path,'dir')
            cmd = sprintf('mkdir %s', ou_path);
            system(cmd);
        end
        
        ou_name = fullfile(ou_path, files(select(j)).name);
        
        imwrite(im, ou_name);
        
    end
    
    for j = 26:100
        in_name = fullfile(path, folders(i).name, files(select(j)).name);
        im = imread(in_name);
        im = im2double(im);
        
        ou_path = fullfile(outputpath_train, folders(i).name);
        if ~exist(ou_path,'dir')
            cmd = sprintf('mkdir %s', ou_path);
            system(cmd);
        end
        
        ou_name = fullfile(ou_path, files(select(j)).name);
        
        imwrite(im, ou_name);
        
    end

end
    
    
    
    
    
    
    