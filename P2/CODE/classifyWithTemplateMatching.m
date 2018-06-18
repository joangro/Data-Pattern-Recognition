function [ estimatedLabels ] = classifyWithTemplateMatching( templates , testData , errorMeasure,emotions)
%CLASSIFYWITHTEMPLATEMATCHING Given a set of templates and a test dataset,
%this function estimates the labels of each sample in the test dataset
%comparing it with each of the templates.
   

    %init the variable where the estimated labels will be stored
    estimatedLabels = zeros(1,size(testData,1));
    %get the number of templates we are going to evaluate
    numTemplates = size(templates,1);
    
    %Iterate over all the test data
    for i = 1:size(testData,1)
        %get the current sample we want to evaluate
        currentSample = squeeze(testData(i,:,:));
        %init the similarity score for each template with the current
        %sample
        templateScore = zeros(1,numTemplates);
        for e = 1:numTemplates
            %get the current template
            currentTemplate = squeeze(templates(e,:));
            %get the similarity score of the pattern with the given sample
            %and store into templateScore variable
            switch errorMeasure
                case 'euclidean'
                    templateScore(e) = pdist2(currentSample(:)', currentTemplate(:)','euclidean');
                case 'gabor-dist'
                    [MAG, ~] = imgaborfilt(reshape(currentSample, 128,128 ),2,90);
                    templateScore(e) = pdist2(MAG(:)', currentTemplate(:)','euclidean');
                case 'std-dist'
                    [MAG] = stdfilt(reshape(currentSample, 128,128 ));
                    templateScore(e) = pdist2(MAG(:)', currentTemplate(:)','euclidean');
                case 'fib-dist'  
                   
                    MAG = fibermetric( reshape(currentSample, 128,128 ));
                    templateScore(e) = pdist2(MAG(:)', currentTemplate(:)','euclidean');


            end
        end        
        %get the label with the minimum similarity score and assign it to
        %the current sample
        estimatedLabels(i) = emotions(find(templateScore==min(templateScore),1));
        
    end
end

