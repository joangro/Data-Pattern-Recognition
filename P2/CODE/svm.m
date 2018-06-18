clear variables
close all

%%%% SVM

%% Choose the emotion labels we want to classify in the database
% 0:Neutral 
% 1:Angry 
% 2:Bored 
% 3:Disgust 
% 4:Fear 
% 5:Happiness 
% 6:Sadness 
% 7:Surprise
emotionsUsed = [1 6];  

%%%%%%%%%%%%%%%% EXTRACT DATA %%%%%%%%%%%%
[imagesData shapeData labels stringLabels] = extractData('CKDB', emotionsUsed);

%SVMStruct = svmtrain(Training,Group);

%%%%%%%%%%%%%%%% EXTRACT FEATURES %%%%%%%%%%%%
grayscaleFeatures = extractFeaturesFromData(imagesData,'grayscale');
K = 6;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);
train = grayscaleFeatures(indexesCrossVal~=1,:,:);
test = grayscaleFeatures(indexesCrossVal==1,:,:);
labels = labels';
labels = labels(indexesCrossVal~=1,:,:);
SVMStruct = svmtrain(train, labels,  'showplot',true);
Group = svmclassify(SVMStruct,test,'ShowPlot',true);
