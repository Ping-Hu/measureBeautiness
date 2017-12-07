% This script cooks data for the CNN model.
% Now we have 20 categories of faces, 44000 images in each category.
%--------------------------------------------------------------------------
% TRAINING SET:
% We randomly select 11000 images from every category to compose our
% training set. In total, we have 20*11000 = 220,000 images in the training
% set.
%--------------------------------------------------------------------------
% VALIDATION SET:
% Then, among the unselected images, we randomly collect 5500 images from
% each category as the validation set. In total, we have 20*5500 = 110,000
% images in the validation set.
%--------------------------------------------------------------------------
% TESTING SET:
% In the rest of images, we randomly select 5500 images from each category
% to compose the testing set. In total, we have 20*5500 = 110,000
% images in the testing set.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Implementation:
% step1. generate 22000 (11000+5500+5500) images random numbers in the range of
%        [1:44000].
% step2. Get the first 11000 images as the training set,
%        the next 5500 images as the validation set,
%        and the last 5500 images as the testing set.
% step3. To build up 20 batches for training set:
%        In every batch, we should have 11000 images. Randomly generate 19
%        numbers in the range of [1, 11000]. The resulting 20 intervals
%        will be the amount of images we should take from every category.
%        Putting this composition of images into one batch. Meanwhile, keep
%        the corresponding label of the image in another .dat file called
%        labels. Repeat this procedure for the 20 batches.
%        After that, we get the batches of training set.
% step4. To build up 10 batches for validation set:
%        Similar procedure as step3 excepting the range is but [1, 5500].
% step5. To build up 10 batches for testing set:
%        Similar procedure as step4.
% Cooking data is completed.
%--------------------------------------------------------------------------
clear all;

% amount of training images in one category
trNumber = 2000;
% amount of validating images in one category
vaNumber = 500;
% amount of testing images in one category
teNumber = 500;
% image size
height = 100;
width = 75;

inputdir = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/FL/';
outputdir = '/Users/Peggy/Documents/MATLAB/FinalProject/dataset_face/preprocess/batchdata/';
category = dir(inputdir);
%category = [];
% amount of category
while ~isempty(category(1).name)
    if (category(1).name(1) == '.')
        category(1) = [];
    else
        break;
    end
end
% amount of categories
caNumber = length(category);
% to remember indices in every category
caAllIndex = struct('categoryID',[],...
    'trainIndex',[],...
    'validIndex',[],...
    'testIndex',[],...
    'filelist',[]);

for i = 1:caNumber
    
    
    path = fullfile(inputdir,'/',category(i).name,'/');
    filelist = dir(path);
    while ~isempty(filelist(1).name)
        if (filelist(1).name(1) == '.')
            filelist(1) = [];
        else
            break;
        end
    end
    for j = 1: length(filelist)
        fullname = fullfile(path,filelist(j).name);
        filelist(j).name = fullname;
    end
    
    totalNumber = trNumber+vaNumber+teNumber;
    selected = randperm(totalNumber, totalNumber);
    % selected has only one row
    trIndex = selected(1:trNumber);
    vaIndex = selected(trNumber+1: trNumber+vaNumber);
    teIndex = selected(trNumber+vaNumber+1: totalNumber);
    if isempty(caAllIndex(end).categoryID)
        caAllIndex(end).categoryID = category(i).name;
        caAllIndex(end).trainIndex = trIndex;
        caAllIndex(end).validIndex = vaIndex;
        caAllIndex(end).testIndex  = teIndex;
        caAllIndex(end).filelist = filelist;
    else
        caAllIndex(end+1).categoryID = category(i).name;
        caAllIndex(end).trainIndex = trIndex;
        caAllIndex(end).validIndex = vaIndex;
        caAllIndex(end).testIndex  = teIndex;
        caAllIndex(end).filelist = filelist;
        
    end
end


trbaIndex = struct('batchID',[],'amount',0,'filelist',[],'labels',[]);
vabaIndex = struct('batchID',[],'amount',0,'filelist',[],'labels',[]);
tebaIndex = struct('batchID',[],'amount',0,'filelist',[],'labels',[]);
trbaNumber = 4;
vabaNumber = 1;
tebaNumber = 1;

% for all the 20 batches:

%--------------------------------------------------------------------------
% for the training set
%--------------------------------------------------------------------------

trainbatchIndex = randperm(caNumber*trNumber,caNumber*trNumber);

for i = 1:trbaNumber
    
    amount = trNumber*caNumber/trbaNumber;
    
    starting = (i-1)*amount+1;
    ending = i*amount;
    
    thisbaIndex = trainbatchIndex(starting:ending);
    %     thiscaIndex = thiscaIndex-(i-1)*amount;
    
    trfl = struct('filename',[]);
    trlb = struct('la',[]);
    for j = 1:amount
        % eg: categoryID = mod(55395,11000);
        
        cinTr = mod(thisbaIndex(j),trNumber);
        if cinTr == 0
            cinTr = trNumber;
            cid = floor(thisbaIndex(j)/trNumber);
        else
            %cinTr = cinTr;
            cid = floor(thisbaIndex(j)/trNumber)+1;
        end
        
        %       fprintf(int2str(cid),'/n');
        %thisCa = struct();
        
        realid = str2num( findCaID(caNumber, caAllIndex, cid));
        %         for m = 1:caNumber
        %             if (caAllIndex(m).categoryID == cid)
        %                 realid = caAllIndex(m).categoryID;
        %
        %             end
        %         end
        realin = caAllIndex(realid).trainIndex(cinTr);
        thislist = caAllIndex(realid).filelist;
        %         realid = thisCa.categoryID;
        
        %         realin = caAllIndex(realid).trainIndex(cinTr);
        %thislist = caAllIndex(cid).filelist;
        
        %             if isempty(fl(end).filename)
        %                 fl(end).filename = thislist(cin).name;
        %             else
        %                 fl(end+1).filename = thislist(cin).name;
        %             end
        trfl(j).filename = thislist(realin).name;
        
        %             if isempty(lb(end).la)
        %                 lb(end).la = realid;
        %             else
        %                 lb(end+1).la = realid;
        %             end
        trlb(j).la = realid;
        
    end
    
    if isempty(trbaIndex(end).batchID)
        trbaIndex(end).batchID = i;
        trbaIndex(end).amount = amount;
        trbaIndex(end).filelist = trfl;
        trbaIndex(end).labels = trlb;
    else
        
        trbaIndex(end+1).batchID = i;
        trbaIndex(end).amount = amount;
        trbaIndex(end).filelist = trfl;
        trbaIndex(end).labels = trlb;
        
    end
    
    
    batchdata = zeros(amount,height*width*3);
    batchlabels = zeros(amount,1);
    if amount ~= length(trbaIndex(end).filelist)
        fprintf('Number of training files in one batch is not the same as the number in the filelist!');
    end
    
    for j = 1:amount
        thisfname = trbaIndex(end).filelist(j).filename;
        im = imread(thisfname);
        [r,c,d] = size(im);
        r = r-25;
        
        for t = 1:d
            for s = 1:c
                for q = 1:r
                    
                    batchdata(j,c*r*(t-1)+(s-1)*r+q) = im(q,s,t);
                end
            end
        end
        batchlabels(j) = trbaIndex(end).labels(j).la;
    end
    
    save(strcat('trainbatch',int2str(trbaIndex(i).batchID),'.mat'),'batchdata');
    save(strcat('trainlabels',int2str(trbaIndex(i).batchID),'.mat'),'batchlabels');
    
end


%--------------------------------------------------------------------------
% for the validation set
%--------------------------------------------------------------------------

validbatchIndex = randperm(caNumber*vaNumber,caNumber*vaNumber);

for i = 1:vabaNumber
    
    amount = vaNumber*caNumber/vabaNumber;
    
    starting = (i-1)*amount+1;
    ending = i*amount;
    
    thisbaIndex = validbatchIndex(starting:ending);
    %     thiscaIndex = thiscaIndex-(i-1)*amount;
    
    vafl = struct('filename',[]);
    valb = struct('la',[]);
    for j = 1:amount
        % eg: categoryID = mod(55395,11000);
        
        cinVa = mod(thisbaIndex(j),vaNumber);
        if cinVa == 0
            cinVa = vaNumber;
            cid = floor(thisbaIndex(j)/vaNumber);
        else
            %cinTr = cinTr;
            cid = floor(thisbaIndex(j)/vaNumber)+1;
        end
        
        %       fprintf(int2str(cid),'/n');
        realid = str2num( findCaID(caNumber, caAllIndex, cid));
        
        %         for m = 1:caNumber
        %             if caAllIndex(m).categoryID == cid
        %                 realid = caAllIndex(m).categoryID;
        %                 realin = caAllIndex(realid).validIndex(cinVa);
        %                 thislist = caAllIndex(realid).filelist;
        %             end
        %         end
        %
        realin = caAllIndex(realid).testIndex(cinVa);
        thislist = caAllIndex(realid).filelist;
        %         realid = thisCa.categoryID;
        %         thislist = thisCa.filelist;
        %         realin = caAllIndex(realid).validIndex(cinVa);
        %thislist = caAllIndex(cid).filelist;
        
        %             if isempty(fl(end).filename)
        %                 fl(end).filename = thislist(cin).name;
        %             else
        %                 fl(end+1).filename = thislist(cin).name;
        %             end
        vafl(j).filename = thislist(realin).name;
        
        %             if isempty(lb(end).la)
        %                 lb(end).la = realid;
        %             else
        %                 lb(end+1).la = realid;
        %             end
        valb(j).la = realid;
        
    end
    
    if isempty(vabaIndex(end).batchID)
        vabaIndex(end).batchID = i;
        vabaIndex(end).amount = amount;
        vabaIndex(end).filelist = vafl;
        vabaIndex(end).labels = valb;
    else
        
        vabaIndex(end+1).batchID = i;
        vabaIndex(end).amount = amount;
        vabaIndex(end).filelist = vafl;
        vabaIndex(end).labels = valb;
        
    end
    
    
    batchdata = zeros(amount,height*width*3);
    batchlabels = zeros(amount,1);
    if amount ~= length(vabaIndex(end).filelist)
        fprintf('Number of validation files in one batch is not the same as the number in the filelist!');
    end
    
    for j = 1:amount
        thisfname = vabaIndex(end).filelist(j).filename;
        im = imread(thisfname);
        [r,c,d] = size(im);
        r = r-25;
        
        for t = 1:d
            for s = 1:c
                for q = 1:r
                    
                    batchdata(j,c*r*(t-1)+(s-1)*r+q) = im(q,s,t);
                end
            end
        end
        batchlabels(j) = vabaIndex(end).labels(j).la;
    end
    
    save(strcat('validbatch',int2str(vabaIndex(i).batchID),'.mat'),'batchdata');
    save(strcat('validlabels',int2str(vabaIndex(i).batchID),'.mat'),'batchlabels');
    
end

%--------------------------------------------------------------------------
% for the testing set
%--------------------------------------------------------------------------
testbatchIndex = randperm(caNumber*teNumber,caNumber*teNumber);

for i = 1:tebaNumber
    
    amount = teNumber*caNumber/tebaNumber;
    
    starting = (i-1)*amount+1;
    ending = i*amount;
    
    thisbaIndex = testbatchIndex(starting:ending);
    %     thiscaIndex = thiscaIndex-(i-1)*amount;
    
    tefl = struct('filename',[]);
    telb = struct('la',[]);
    for j = 1:amount
        % eg: categoryID = mod(55395,11000);
        
        cinTe = mod(thisbaIndex(j),teNumber);
        if cinTe == 0
            cinTe = teNumber;
            cid = floor(thisbaIndex(j)/teNumber);
        else
            %cinTr = cinTr;
            cid = floor(thisbaIndex(j)/teNumber)+1;
        end
        
        %       fprintf(int2str(cid),'/n');
        realid = str2num( findCaID(caNumber, caAllIndex, cid));
        
        %         for m = 1:caNumber
        %             if caAllIndex(m).categoryID == cid
        %                realid = caAllIndex(m).categoryID;
        %                realin = caAllIndex(realid).testIndex(cinTe);
        %                thislist = caAllIndex(realid).filelist;
        %             end
        %         end
        realin = caAllIndex(realid).testIndex(cinTe);
        thislist = caAllIndex(realid).filelist;
        %         realid = thisCa.categoryID;
        %         thislist = thisCa.filelist;
        %         realin = caAllIndex(realid).testIndex(cinTe);
        %thislist = caAllIndex(cid).filelist;
        
        %             if isempty(fl(end).filename)
        %                 fl(end).filename = thislist(cin).name;
        %             else
        %                 fl(end+1).filename = thislist(cin).name;
        %             end
        tefl(j).filename = thislist(realin).name;
        
        %             if isempty(lb(end).la)
        %                 lb(end).la = realid;
        %             else
        %                 lb(end+1).la = realid;
        %             end
        telb(j).la = realid;
        
    end
    
    if isempty(tebaIndex(end).batchID)
        tebaIndex(end).batchID = i;
        tebaIndex(end).amount = amount;
        tebaIndex(end).filelist = tefl;
        tebaIndex(end).labels = telb;
    else
        
        tebaIndex(end+1).batchID = i;
        tebaIndex(end).amount = amount;
        tebaIndex(end).filelist = tefl;
        tebaIndex(end).labels = telb;
        
    end
    
    
    batchdata = zeros(amount,height*width*3);
    batchlabels = zeros(amount,1);
    if amount ~= length(tebaIndex(end).filelist)
        fprintf('Number of training files in one batch is not the same as the number in the filelist!');
    end
    
    for j = 1:amount
        thisfname = tebaIndex(end).filelist(j).filename;
        im = imread(thisfname);
        [r,c,d] = size(im);
        r = r-25;
        
        for t = 1:d
            for s = 1:c
                for q = 1:r
                    
                    batchdata(j,c*r*(t-1)+(s-1)*r+q) = im(q,s,t);
                end
            end
        end
        batchlabels(j) = tebaIndex(end).labels(j).la;
    end
    
    save(strcat('testbatch',int2str(tebaIndex(i).batchID),'.mat'),'batchdata');
    save(strcat('testlabels',int2str(tebaIndex(i).batchID),'.mat'),'batchlabels');
    
end


