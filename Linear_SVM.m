% 
% Contains three elements: first- initial interval_size and second - threshold_cutoff
% and third - lamda. If this cell is empty, the defaults are:
%      interval_size      = 0.01;
%      threshold_cutoff = 0.001;
%      lamda         = 10;
%
% Out:
% - models - contains learnt models
%  

function [models] = SVM_classifier( varargin )
  l = length(varargin);
  if l < 3
    fprintf('syntax: SVM_classifier(training_dataset,trainLabels, parameters)\n');
    return;
  elseif l > 3
    fprintf('syntax: SVM_classifier(training_dataset, trainLabels, parameters)');
    return;
  end
  training_dataset    = varargin{1};
  training_dataset    = [ones(size(training_dataset,1),1) training_dataset];
  trainLabels = varargin{2};
  classifierParameters = varargin{3};
  training_number      = size(training_dataset,1);
  feature_numbers   = size(training_dataset,2);
  label_list     = size(trainLabels,2);
  if length(classifierParameters) == 0
      interval_size      = 0.01;
      threshold_cutoff = 0.001;
      lamda         = 10;
  else
      interval_size      = classifierParameters{1};
      threshold_cutoff = classifierParameters{2};
      lamda         = classifierParameters{3};
  end

  if training_number == 0
    %metadeta
    models = {}; return;
  end
    
  labels_sorted = sort(unique(trainLabels));
  nClasses          = length(labels_sorted);
  weights = zeros(feature_numbers,nClasses);
  nIterations=0;
  preLogL = -Inf;
  
  while 1
      
      tmp=exp(training_dataset*weights);
      dsum=sum(tmp,2);
  
      py_k=tmp ./ repmat(dsum,1,nClasses);
      delta=repmat(trainLabels,1,nClasses)==repmat(labels_sorted',training_number,1);
      
      stepk=zeros(size(weights));
      for k=1:(nClasses)
          errork=delta(:,k)-py_k(:,k);
          stepk(:,k)=sum(training_dataset .* repmat(errork,1,feature_numbers), 1)';
      end
      
      weights=weights + interval_size*(stepk - lamda*weights);
      
      logL = likelihood_estimatn(py_k,delta);
      if ~isnan(logL)
          if logL > preLogL + threshold_cutoff
              preLogL = logL;
          elseif interval_size > 0.00001
              interval_size = interval_size/2;
              preLogL = logL;
          else
              break;
          end
      else
          interval_size = interval_size / 10;
          weights = zeros(feature_numbers,nClasses);
          nIterations = 0;
      end
      nIterations = nIterations + 1;
  end
  models = cell(nClasses+1,1);
  models{nClasses+1} = weights;
  
  training_set_metadata.nExamples = training_number;
  training_set_metadata.feature_numbers = feature_numbers;
  training_set_metadata.label_list = label_list;
  training_set_metadata.nClasses = nClasses;
  training_set_metadata.labels_sorted = labels_sorted;
  training_set_metadata.classPriors = zeros(nClasses,1);
  for l=1:nClasses
    training_set_metadata.classPriors(l) = length(find(trainLabels==labels_sorted(l)));
  end
  training_set_metadata.classPriors = training_set_metadata.classPriors/training_number;   
  
  models{nClasses+2} = training_set_metadata;
  models{nClasses+3} =[];
  
  
function [L] = likelihood_estimatn(py_k,delta)
    L_examples = log( sum(delta .* py_k,2) );
    L = sum(L_examples,1);
end