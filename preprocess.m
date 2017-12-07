% this is the preprocessing part of the training
%--------------------------------------------------------------------------
% Input
% outPath: output file path, optional. default is the same as input path;
% constep: contrast change per step, recommended: 0.05;
% brightness: when = -0.5, automatically change the brightness
% winR, winC: size of the soft window
% wstrideR: soft window's stride of rows, 
% wstrideC: soft window's stride of columns
% Output
% produced images will be stored in outPath
%--------------------------------------------------------------------------
% Steps:
% S1. adjust contrast and brightness
% S2. apply soft window on the original images
% S3. do flipping of every image
%--------------------------------------------------------------------------

% % S1. adjust contrast and brightness
% constep = 0.05;
% brightness = -0.5;
% inputPath = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/raw2000/faces/';
% outputPath = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/CB/';
% contrast_batch(inputPath, outputPath, constep, brightness);

% S2. apply soft window on the original images
winR = 100;
winC = 75;
wstrideR = 5;
wstrideC = 5;
inputPath = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/CB/';
outputPath = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/SW/';
softwindow_batch(inputPath, outputPath, winR, winC, wstrideR, wstrideC);

% S3. do flipping of every image
inputPath = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/SW/';
outputPath = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/FL/';
flipping_batch(inputPath, outputPath);



