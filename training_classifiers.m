% training_classifiers
%
% Trains a classifier on a set of examples
%
% This code wraps around the code for several different classifiers
% and knows how to package their results in a uniform manner, to
% pass to applyClassifier.m

% Example:
%   classifier = training_classifiers(examples, labels, 'kNN', {0.01 0.001 10});

function [trainedClassifier] = training_classifiers( varargin )

l = length(varargin);
if l < 3; help training_classifiers; return; end
  
examples       = varargin{1};
labels         = varargin{2};
selected_classifier = varargin{3};

classifierParameters = {};
if l > 3; classifierParameters = varargin{4}; end
sorted_labels      = sort(unique(labels));
nClasses              = length(sorted_labels);
[numTrain,numFeatures] = size(examples);

for l = 1:nClasses
  label = sorted_labels(l);
  examplesWithLabel{l} = find(labels == label);
end

models = cell(nClasses+3,1);
lcp = length(classifierParameters);

switch selected_classifier
  
 case {'kNN'}
  fprintf('training_classifiers: using %s with parameters\n',selected_classifier); 
  [models] = kNN_classifier(examples,labels,classifierParameters);
  
 case {'SVM'}
  fprintf('training_classifiers: using %s with parameters\n',selected_classifier);
  [models] = linear_SVM(examples,labels,classifierParameters);
  
 case {'nbayes','nbayesPooled','lda','qda','ldaSVD','ldaCV','qdaSVD','qdaCV','nbayes-unitvariance','nbayesPooledResampling'}
    [models] = Gaussian_Naive_Bayes(examples,labels,selected_classifier,classifierParameters);

 case {'knn'}
  k = 1;
  distance = 'Euclidean';

  if lcp > 0
    % override defaults
    k = classifierParameters{1};
    if lcp > 1
      distance = classifierParameters{2};
    end
  end
  
  models{nClasses+1} = cell(3,1);
  models{nClasses+1}{1} = examples;
  models{nClasses+1}{2} = labels;
  models{nClasses+1}{3} = k;
  models{nClasses+1}{4} = distance;

case {'neural'}
  net = classifierNeuralNetwork(examples,labels,classifierParameters);
  models{nClasses+1} = net;
  
  allPairModels = cell(nClasses,nClasses);
  
  fprintf('pairwise classifier with %s\n',classifierToUse);
  
  for l1 = 1:nClasses-1
    for l2 = l1+1:nClasses
      
      c1 = sorted_labels(l1);
      c2 = sorted_labels(l2);
      fprintf('\ttraining %d %d\n',c1,c2);
      
      indices      = find((labels==c1)|(labels==c2));
      pairExamples = examples(indices,:);
      pairLabels   = labels(indices,:);
      allPairModels{l1,l2} = training_classifiers(pairExamples,pairLabels,classifierToUse,classifierParametersToUse);
    end;
  end;

case {'pairwise'}
  classifierToUse           = classifierParameters{1};
  classifierParametersToUse = {};
  if length(classifierParametersToUse) > 1
    classifierParametersToUse = classifierParameters{2};
end

  models{nClasses+1} = allPairModels;

 case {'nnets'}
  [models] = classifierNNets(examples,labels);

  case {'svmlight'}
  fprintf('training_classifiers: using %s with parameters\n',selected_classifier);
  if lcp
    kernel       = classifierParameters{1}; % kernel type
    kernelParams = classifierParameters{2}; % a vector of parameter values    
  else
    % default to a linear kernel
    kernel = 0;
    kernelParams = [];
  end
  net = svml('tmpsvm', 'Kernel', kernel , 'KernelParam', kernelParams);
    
  % convert labels to +1 or -1
  if nClasses > 2
    return;
  else
    indices1 = find(labels == sorted_labels(1));
    indices2 = find(labels == sorted_labels(2));
    labels(indices1) = -1;
    labels(indices2) = 1;
  end
    
  models{nClasses+1} = svmltrain(net, examples, labels);
    
 case {'svm'}
  params = size(classifierParameters,2);
  ker = 'linear';
  if (params > 0); ker = classifierParameters{1};
  end;
  C = inf; 
  if (params > 1); C = classifierParameters{2}; end;
  global p1;
  p1 = 1;
  if (params > 2); p1 = classifierParameters{3};end;
  global p2;
  p2 = 0;
  if (params > 3); p2 = classifierParameters{4}; end;
  correctedTrainLabels = zeros(size(labels));
  indices = find(labels == sorted_labels(1));
  correctedTrainLabels(indices) = -1;
  indices = find(labels == sorted_labels(2));
  correctedTrainLabels(indices) = 1;
  
  [nsv,alpha,bias] = svc(examples,correctedTrainLabels,ker,C);
  
  models = cell(nClasses+3,1);
  models{nClasses+1} = cell(5,1);
  models{nClasses+1}{1} = examples;
  models{nClasses+1}{2} = labels;
  models{nClasses+1}{3} = ker;
  models{nClasses+1}{4} = alpha;
  models{nClasses+1}{5} = bias;
  
 otherwise
end

switch selected_classifier
 case {'knn','svmlight','pairwise','neural','nnets','svm','kNN'}

  % Training Set information
  training_set_metadata.classifierParameters = classifierParameters;
  training_set_metadata.nExamples            = numTrain;
  training_set_metadata.nFeatures            = numFeatures;
  training_set_metadata.nClasses             = nClasses;
  training_set_metadata.sorted_labels    = sorted_labels;
  training_set_metadata.classPriors          = zeros(nClasses,1);
  for l=1:nClasses
    training_set_metadata.classPriors(l) = length(find(labels==sorted_labels(l)));
  end
  training_set_metadata.classPriors = training_set_metadata.classPriors/numTrain;            
  
 otherwise
  training_set_metadata = models{nClasses+2};
end
training_set_metadata.examplesWithLabel = examplesWithLabel;
models{nClasses+2} = training_set_metadata;
trainedClassifier.models          = models;
trainedClassifier.training_set_metadata = training_set_metadata;
trainedClassifier.when            = datestr(now);
trainedClassifier.classifier      = selected_classifier;
trainedClassifier.classifierParameters      = classifierParameters;
training_set_metadata;