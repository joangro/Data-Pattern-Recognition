clear variables
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% P2 - RECONEIXEMENT DE PATRONS  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%    REDUCCI� DE DIMENSIONALITAT %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PLOT YES(1)/NO(0)
plot_graphs = 0;

%% Calculate overall error
calculate_quad_err = 0;

%% Choose the emotion labels we want to classify in the database
% 0:Neutral 
% 1:Angry 
% 2:Bored 
% 3:Disgust 
% 4:Fear 
% 5:Happiness 
% 6:Sadness 
% 7:Surprise
emotionsUsed = [1 2 6];  

%%%%%%%%%%%%%%%% EXTRACT DATA %%%%%%%%%%%%
[imagesData shapeData labels stringLabels] = extractData('CKDB', emotionsUsed);

%%%%%%%%%%%%%%%% EXTRACT FEATURES %%%%%%%%%%%%
grayscaleFeatures = extractFeaturesFromData(imagesData,'fib-filter');


%%GSCATTER 3 example. Visualize the first three coordiantes of the data.
%%You can remove this after understanding it! :)
if plot_graphs==1
    figure(1);
    gscatter3(grayscaleFeatures(:,1),grayscaleFeatures(:,2),grayscaleFeatures(:,3),stringLabels,7)
    title('Original Data')
end

dimensions = 200;
[dataProjected meanProjection vectorsProjection] = reduceDimensionality( grayscaleFeatures, 'PCA', dimensions, labels  );

RZ_mean_proj = reshape(meanProjection, 128, 128);
% RZ_vector_proj = reshape(vectorsProjection, 128, 128,3); <- no serveix
if plot_graphs==1
    figure(2);
    imagesc(RZ_mean_proj)
    title('Mean projection')
end

[ dataReprojected ] = reprojectData( dataProjected , meanProjection, vectorsProjection );
if plot_graphs==1
    figure(3);
    imagesc(reshape(dataReprojected(1,:,:), 128,128 ));
    title('Reprojected image' )
    xlabel(['Dimensionality:  ' num2str(dimensions)])
end

% Calculate quadratic error from whole database
if calculate_quad_err == 1

    dimensions = [2, 5, 10, 50, 100, 300, 500];
    err = zeros(1, length(dimensions));
    for i = 1:length(dimensions)
        [dataProjected meanProjection vectorsProjection] = reduceDimensionality( grayscaleFeatures, 'PCA', dimensions(i), labels  );
        [ dataReprojected ] = reprojectData( dataProjected , meanProjection, vectorsProjection );
        sum_err = 0;
        for j = 1:size(imagesData,1)
            reshape_reprojection = reshape(dataReprojected(j,:,:), 128,128 );
            sum_err = sum_err + immse(squeeze(imagesData(j,:,:)),reshape_reprojection);
        end
        err(i) = sum_err;
    end
    figure(4)
    plot(err, 'r-o')
    ylabel('Error')
    xlabel('Dimensionality: 2, 5, 10, 50, 100, 300, 500')
    title('Quadratic Error')
end
%%%%%%%%%%%%%%% DIVIDE DATA (TRAIN/TEST) WITH CROSS VALIDATION  %%%%%%%%%
K = 6;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);

%%%%%%%  EXAMPLE OF CLASSIFYING THE EXPRESSION USING TEMPLATE  MATCHING %%%%
[ACC CONF ] = testTemplateMatchingWithDR( grayscaleFeatures , labels, emotionsUsed , 'fib-dist', indexesCrossVal )