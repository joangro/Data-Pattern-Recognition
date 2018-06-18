
% JOAN GRAU NOËL - 172578

%% EX 1

clear variables
load ionosphere 
p = .7;    % proportion of rows to select for training
N = size(X,1);  % total number of rows 
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;     
tf = tf(randperm(N));   % randomise order
Xtrain = X(tf,:);
Ytrain = Y(tf,:);
Xtest = X(~tf,:);
Ytest = Y(~tf,:);

% TRAIN
svmModel=fitcsvm(X,Y) %SVM
ldaModel=fitcdiscr(X,Y, 'discrimType', 'diaglinear' ) %LDA

% TEST
svmPrediction = predict(svmModel, Xtest)
ldaPrediction = predict(ldaModel, Xtest)

% ACCURACY
acc_svm = mean(strcmp(Ytest,svmPrediction))
acc_svm = mean(strcmp(Ytest,ldaPrediction))

%% EX 2

% LOAD AND VISUALIZE
clear variables
close all
load spectra
whos NIR octane

X=NIR;
Y=octane;
[dummy,h] = sort(octane);
oldorder = get(gcf,'DefaultAxesColorOrder');
set(gcf,'DefaultAxesColorOrder',jet(60));
plot3(repmat(1:401,60,1)',repmat(octane(h),1,401)',NIR(h,:)');
set(gcf,'DefaultAxesColorOrder',oldorder);
xlabel('Wavelength Index'); ylabel('Octane'); axis('tight');
grid on

%MR
betaMR=regress(Y-mean(Y),X); 

%PLS
ndims = 5;
[ ~ ,~ ,~ ,~ ,betaPLS]=plsregress(X,Y,ndims);  

% PCR
[PCALoadings,PCAScores,PCAVar] = pca(X,'Economy',false);
betaPCR = regress(Y-mean(Y), PCAScores(:,1:2));
betaPCR = PCALoadings(:,1:2)*betaPCR;
betaPCR = [mean(Y) - mean(X)*betaPCR; betaPCR];


% Predict
yPredMR= X*betaMR;
yPredPCR = [ones(size(X,1),1) X]*betaPCR;
yPredPLS=[ones(size(X,1),1) X]*betaPLS;

% Error
errorMR = (1/60)*sum(sum(abs(yPredMR-Y)/Y))
errorPCR = (1/60)*sum(sum(abs(yPredPCR-Y)/Y))
errorPLS = (1/60)*sum(sum(abs(yPredPLS-Y)/Y))
