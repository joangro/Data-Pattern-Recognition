function [ estimatedLabels ] = classifyWithTemplateMatching( templates , testData , method, errorMeasure,emotions)
%CLASSIFYWITHTEMPLATEMATCHING Given a set of templates and a test dataset,
%this function estimates the labels of each sample in the test dataset
%comparing it with each of the templates.
    
    %Convert all the images in the testData into a chamfer distance images
    if(strcmp(method,'chamferMean')==1)
        for i = 1:size(testData,1)
            image = squeeze(testData(i,:,:));
            testData(i,:,:) = bwdist(edge(image,'canny',0.4));
        end    
    end

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
            currentTemplate = squeeze(templates(e,:,:));
            %get the similarity score of the pattern with the given sample
            %and store into templateScore variable
            switch errorMeasure
                case 'euclidean'
                    templateScore(e) = pdist2(currentSample(:)', currentTemplate(:)','euclidean');
                    
                case 'mean-dist'
                    for k = 1:size(currentSample,1)
                        for j = 1:size(currentSample,2)
                            z(k,j) = mean(currentSample(k,j))- currentTemplate(1);
                        end
                    end
                    templateScore(e) = mean(mean(z));
                    
                case 'z-dist'
                    z = (currentSample - currentTemplate(1:128,:))/currentTemplate(129:end,:);
                    templateScore(e) = mean(mean(z));
                    
                   
                case 'z-dist-pixel'
                    for k = 1:size(currentSample,1)
                        for j = 1:size(currentSample,2)
                            z(k,j) = ((currentSample(k,j) - currentTemplate(k,j))^2)/currentTemplate(i+128,j);
                        end
                    end
                    templateScore(e) = mean(mean(z));
                    
                case 'hist-dist'
                    histograma = histcounts(currentSample,50);
                    templateScore(e) = mean(mean(histograma-currentTemplate).^2);
                    
                case 'gabor-dist'
                    [MAG, ~] = imgaborfilt(currentSample,2,90);
                    templateScore(e) = pdist2(MAG(:)', currentTemplate(:)','euclidean');
                    
                case 'z-gabor-dist'
                    [MAG, ~] = imgaborfilt(currentSample,2,90);
                    templateScore(e) = mean(mean((MAG - currentTemplate(1:128,:))./currentTemplate(129:end,:)));
                    
                    %% FROM HERE THEY ARE ALL IMAGE FILTERS
                    % First there is the euclidean distance, then the
                    % Z-dist for each filter
                case 'std-dist'
                    [MAG] = stdfilt(currentSample);
                    templateScore(e) = pdist2(MAG(:)', currentTemplate(:)','euclidean');

                case 'z-std-dist'
                    [MAG] = stdfilt(currentSample);
                    templateScore(e) = mean(mean((MAG - currentTemplate(1:128,:))./currentTemplate(129:end,:)));
                    
                case 'range-dist'
                    [MAG] = rangefilt(currentSample);
                    templateScore(e) = pdist2(MAG(:)', currentTemplate(:)','euclidean');

                case 'z-range-dist'
                    MAG = rangefilt(currentSample);
                    templateScore(e) = mean(mean((MAG - currentTemplate(1:128,:))./currentTemplate(129:end,:)));

                case 'fib-dist'  
                    MAG = fibermetric(currentSample);
                    templateScore(e) = pdist2(MAG(:)', currentTemplate(:)','euclidean');

                case 'z-fib-dist'  
                    MAG = fibermetric(currentSample);
                    templateScore(e) = mean(mean((MAG - currentTemplate(1:128,:))./currentTemplate(129:end,:)));
                    

            end
            
        end        
        %get the label with the minimum similarity score and assign it to
        %the current sample
        estimatedLabels(i) = emotions(find(templateScore==min(templateScore),1));
        
    end
end

