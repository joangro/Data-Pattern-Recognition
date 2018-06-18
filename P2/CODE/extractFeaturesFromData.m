function [ features ] = extractFeaturesFromData( data , featureType )
%EXCTRACTFEATURESFROMDATA Summary of this function goes here
%   Detailed explanation goes here
    switch featureType
        case 'grayscale'
            features = reshape(data,size(data,1),128*128);
        case 'gabor'
          MAG = zeros(size(data));
          for i = 1:size(data,1)
            [MAG(i,:,:), ~] = imgaborfilt(squeeze(data(i,:,:)),2,90);
          end
          features = reshape(MAG,size(data,1),128*128);
        case 'std-filter'
              MAG = zeros(size(data));
              for i = 1:size(data,1)
                MAG(i,:,:) = stdfilt(squeeze(data(i,:,:)));
              end
          features = reshape(MAG,size(data,1),128*128);
         case 'fib-filter'
              MAG = zeros(size(data));
              for i = 1:size(data,1)
                MAG(i,:,:) = fibermetric(squeeze(data(i,:,:)));
              end
          features = reshape(MAG,size(data,1),128*128);

    end
end

