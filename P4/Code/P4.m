close all
clear variables
%%
imDatasetPath = fullfile(pwd, 'CKDB');
imData = imageDatastore(imDatasetPath, 'IncludeSubfolders',true,'LabelSource','foldernames');

% Use same number of data on all categories
numlabels = countEachLabel(imData);
minSetCount = min(numlabels{:,2}); 
imData = splitEachLabel(imData, minSetCount, 'randomize');

% Split between training and test set
NumTrain = int8(min(numlabels{:,2})/2);
[trainData, labelData] = splitEachLabel(imData,NumTrain,'randomize');

% Create layers

layers = [
    imageInputLayer([128 128 1])

    convolution2dLayer(3,32,'Padding',1)
    reluLayer

    convolution2dLayer(3,64,'Padding',1)
    reluLayer
    
    convolution2dLayer(3,128,'Padding',1)
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,256,'Padding',1)
    reluLayer

    convolution2dLayer(3,128,'Padding',1)
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,256,'Padding',1)
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm');
net = trainNetwork(trainData,layers, options);