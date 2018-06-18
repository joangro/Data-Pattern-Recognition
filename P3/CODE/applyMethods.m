function [ accuracy confusionMatrix ] = applyMethods(data, labels, labelsUsed, indexesCrossVal, classificationMethod,dimensionalityReductionMethod)
    
    % ApplyMethods
    
    % This function estimates the accuracy and confusion over the dataset samples in data by using Cross Validation 
    % the input template matching method and the error measure method.
    
    % INPUTS:
    % data: NxDxD , matrix with the images or shape data of the dataset
    % labels: 1xN vector with the emotion labels for each sample in the matrix data.
    % labelsUsed: labels used to train and classify.
    % templateMethod: string with the method used to generate the template.
    % errorMeasuse: string with the error measure method used to compare the template with the samples.
    % indexesCrossVal: indexes used to perform the performance evaluation with cross validation

    indexes = indexesCrossVal;
    K = max(indexes);

    confusionMatrix = zeros(numel(labelsUsed));

    for k = 1:K
        display(['Testing data subset: ' num2str(k) '/' num2str(K)]);
        %get train and test dataset with the indexes obtained with the KFold
        %cross validation
        trainSamples = data(indexes~=k,:,:);
        labelsTrain  = labels(indexes~=k);
        
        testSamples  = data(indexes==k,:,:);
        labelsTest   = labels(indexes==k);

        switch dimensionalityReductionMethod
            case 'PCA'
                %CODE HERE
                dimensions = 30;
                [ trainSamples_p, meanProjection_train, vectorsProjection_train ] = reduceDimensionality( trainSamples, 'PCA', dimensions,labelsTrain  );
                [ testSamples_p, meanProjection_test, vectorsProjection_test ] = reduceDimensionality( testSamples, 'PCA', dimensions,labelsTest  );
                 trainSamples_r = reprojectData( trainSamples_p , meanProjection_train, vectorsProjection_train );
                 testSamples_r = reprojectData( testSamples_p , meanProjection_test, vectorsProjection_test );
            case 'LDA'
                %CODE HERE
                dimensions = 3;
                [ trainSamples_p, meanProjection_train, vectorsProjection_train ] = reduceDimensionality( trainSamples, 'PCA', dimensions,labelsTrain  );
                [ testSamples_p, meanProjection_test, vectorsProjection_test ] = reduceDimensionality( testSamples, 'PCA', dimensions,labelsTest  );
                [ trainSamples_p, meanProjection_train, vectorsProjection_train ] = reduceDimensionality( trainSamples_p, 'LDA', dimensions,labelsTrain  );
                [ testSamples_p, meanProjection_test, vectorsProjection_test ] = reduceDimensionality( testSamples_p, 'LDA', dimensions,labelsTest  );
                 trainSamples_r = reprojectData( trainSamples_p , meanProjection_train, vectorsProjection_train );
                 testSamples_r = reprojectData( testSamples_p , meanProjection_test, vectorsProjection_test );

            case 'kernelPCA'    
                %CODE HERE
		%Check de compute_mapping function.
            dimensions = 25;
                [ trainSamples_p, meanProjection_train, vectorsProjection_train ] = reduceDimensionality( trainSamples, 'KernelPCA', dimensions,labelsTrain  );
                [ testSamples_p, meanProjection_test, vectorsProjection_test ] = reduceDimensionality( testSamples, 'KernelPCA', dimensions,labelsTest  );
                 trainSamples_r = reprojectData( trainSamples_p , meanProjection_train, vectorsProjection_train );
                 testSamples_r = reprojectData( testSamples_p , meanProjection_test, vectorsProjection_test );

            case 'none'
                trainSamples_r = trainSamples;
                testSamples_r  = testSamples;

        end

        switch classificationMethod
            case 'K-NN'
                % SAMPLE OF MATLAB's implementation of several classifiers
                knn = fitcknn(trainSamples_r, labelsTrain);
                estimatedLabels=knn.predict(testSamples_r);
                
                
            case 'SVM'
                % TODO:
                % Train and classify with an implementation of SVM
                % HINT: check Matlab's svmtrain / svmclassify
                % REMEMBER: basic SVM is intended for binary classification. It MUST be extended to a 
                %  multiclass level, look for a strategy (data partition, iterative one-against-all, etc)
%                 SVMstruct = svmtrain(trainSamples, labelsTrain);
%                 estimatedLabels = svmclassify(SVMstruct, testSamples);
                estimatedLabels = multisvm(trainSamples_r,labelsTrain,testSamples_r, 'linear' );

            case 'Mahalanobis'
%                 testSamplesReshaped = zeros(size(testSamples,1), 128, 128);
%                 for i = 1: size(testSamples,1)
%                     testSamplesReshaped(i,:,:) = reshape(testSamples(i,:,:), 128, 128);
%                     D = mahal( testSamples(i), reshape(meanProjection, 128, 128));
%                 end
%                 estimatedLabels = min(D);
                estimatedLabels = classify(testSamples_r+1e10, trainSamples_r+1e10, labelsTrain', 'Mahalanobis');
            
            case 'kernelSVM-poly'     
                estimatedLabels = multisvm(trainSamples_r,labelsTrain,testSamples_r, 'polynomial');

            case 'kernelSVM-gauss'     
                estimatedLabels = multisvm(trainSamples_r,labelsTrain,testSamples_r, 'rbf');

        end

        %Create confusion matrix evaluating the templates with the test data
        confusionMatrix = confusionMatrix + confusionmat(estimatedLabels, labelsTest, 'ORDER', labelsUsed);
    end
    
    %get the total accuracy of the system
    accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));
end

