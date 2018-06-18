function [ pattern ] = createTemplate( data , namePattern )
%CREATETEMPLATE Given the samples in the data matrix, create a template
%using the namePattern method. 
    switch namePattern
        case 'grayscaleMean'
            %mean of the grayscale images
            pattern = squeeze(mean(data));
            
        case 'gauss-distr' 
            avg = mean(mean(mean(data)));
            var = std(std(std(data)));
            pattern = [ avg; var];
            
        case 'gauss-distr-pixel'
            for i = 1:size(data,2)
                for j = 1:size(data,3)
                    avg(i,j) = mean(data(:,i,j));
                    var(i,j) = std(data(:,i,j));
                end
            end
            pattern = [avg;var];
            
        case 'hist'
            histograma = zeros(size(data,1), 50);
            for i = 1:size(data,1)
                histograma(i,:) = histcounts(data(i,:,:), 50);
            end
            pattern = (mean(histograma,1));
        %% IMAGE FILTERS
        case 'gabor'
          MAG = zeros(size(data));
          for i = 1:size(data,1)
            [MAG(i,:,:), ~] = imgaborfilt(squeeze(data(i,:,:)),2,90);
          end
          pattern = squeeze(mean(MAG,1));
          
        case 'z-gabor'
          MAG = zeros(size(data));
          for i = 1:size(data,1)
            [MAG(i,:,:), ~] = imgaborfilt(squeeze(data(i,:,:)),2,90);
          end
          pattern = squeeze(mean(MAG,1));
          for i = 1:size(data,2)
            for j = 1:size(data,3)
                var(i,j) = (1/size(data,1))*(sum(data(:,i,j)- pattern(i,j)).^2);
            end
          end
          pattern = [pattern; var];    
          %% EXTRA IMAGE FILTERS
          case 'std-filter'
              MAG = zeros(size(data));
              for i = 1:size(data,1)
                MAG(i,:,:) = stdfilt(squeeze(data(i,:,:)));
              end
              pattern = squeeze(mean(MAG,1));
          
          case 'z-std-filter'
              MAG = zeros(size(data));
              for i = 1:size(data,1)
                MAG(i,:,:) = stdfilt(squeeze(data(i,:,:)));
              end
          pattern = squeeze(mean(MAG,1));
          pattern_var = squeeze(std(MAG,1));
          pattern = [pattern; pattern_var];

          case 'range-filter'
              MAG = zeros(size(data));
              for i = 1:size(data,1)
                MAG(i,:,:) = rangefilt(squeeze(data(i,:,:)));
              end
              pattern = squeeze(mean(MAG,1));
         
         case 'z-range-filter'
              MAG = zeros(size(data));
              for i = 1:size(data,1)
                MAG(i,:,:) = rangefilt(squeeze(data(i,:,:)));
              end
              pattern = squeeze(mean(MAG,1));
              pattern_var = squeeze(std(MAG,1));
              pattern = [pattern; pattern_var];

         case 'fib-filter'
              MAG = zeros(size(data));
              for i = 1:size(data,1)
                MAG(i,:,:) = fibermetric(squeeze(data(i,:,:)));
              end
              pattern = squeeze(mean(MAG,1));

         case 'z-fib-filter'
              MAG = zeros(size(data));
              for i = 1:size(data,1)
                MAG(i,:,:) = fibermetric(squeeze(data(i,:,:)));
              end
              pattern = squeeze(mean(MAG,1));
              pattern_var = squeeze(std(MAG,1));
              pattern = [pattern; pattern_var];

    end
end

