% [models] = GNB_classifier( examples, labels, [variation],[parameters] )
%
% Train several flavours of bayesian classifier
% 
% Example:
%  - classifier=GNB_classifier(examples,labels,'nbayes',{})

function [models] = GNB_classifier( varargin )

%
% process arguments
%

l = length(varargin);
if l < 2; help GNB_classifier; return; else
  if l > 2
    classifier = varargin{3};
    params_classifier = {};
    if l > 3; params_classifier = varargin{4}; end
  else
    classifier = 'nbayes';
  end 

  switch classifier
   case {'nbayes','nbayesPooled','nbayes-unitvariance','lda','qda','ldaSVD','ldaCV','qdaSVD','qdaCV','nbayesPooledResampling'}
   otherwise
    fprintf('classifierNBayes: Unknown classifier %s\n',classifier);
  end

end

training_set    = varargin{1};
training_set_labels = varargin{2};

switch classifier
 case {'ldaSVD','ldaCV','qdaSVD','qdaCV'}
  dataPartitionRatio = 0.9;
  if ~isempty(params_classifier); dataPartitionRatio = params_classifier{1};end

  switch classifier
   case {'ldaSVD','qdaSVD'}
    [training_set,V] = Principal_Component_Analaysis(training_set,training_set_labels,'SVD',dataPartitionRatio);
   case {'ldaCV','qdaCV'}
    % first reduce dimensionality
    [training_set,Vsvd] = Principal_Component_Analaysis(training_set,training_set_labels,'SVD',dataPartitionRatio);
    [training_set,Vcv]  = Principal_Component_Analaysis(training_set,training_set_labels,'CV',dataPartitionRatio);
  end  
  
  % crop?
  
 otherwise
end

nTrain      = size(training_set,1);
nFeatures   = size(training_set,2);
nLabels     = size(training_set_labels,2);
sortedLabelValues = sort(unique(training_set_labels));
nClasses          = length(sortedLabelValues);
weights   = zeros(nFeatures+1,nClasses);
models = cell(nClasses+3,1); % will keep the final results

%
% Train
%

means = zeros(nClasses,nFeatures);
stds  = zeros(nClasses,nFeatures);
  
% training also involves learning classPriors (#classes)
classPriors = zeros(nClasses,1);
nPerClass   = zeros(nClasses,1);


%% a) Find the examples belonging to each class and their means/stdevs

indices = cell(nClasses,1);

% hack - save it now, as the training data will have its mean subtracted
switch classifier
 case {'nbayesPooledResampling'}
  models{nClasses+3}{1} = training_set;
  models{nClasses+3}{2} = training_set_labels;
end


for c = 1:nClasses
  cond = sortedLabelValues(c);
  indices{c}   = find( training_set_labels == cond );
  nPerClass(c) = length(indices{c});

  % compute means and stdevs per class
  means(c,:) = mean(training_set(indices{c},:),1);

  switch classifier
   case {'nbayes'}
    stds(c,:)  = std(training_set(indices{c},:),0,1);
    
   case {'nbayes-unitvariance'}
    stds(c,:)  = ones(1,nFeatures);
    
   case {'nbayesPooled','nbayesPooledResampling','lda','qda','ldaSVD','ldaCV','qdaSVD','qdaCV'}
    % subtract means from data, as it will help
    % compute pooled feature standard deviations
    % or a full covariance matrix
    
    training_set(indices{c},:) = training_set(indices{c},:) - repmat(means(c,:),nPerClass(c),1);
  end
end

classPriors = nPerClass / nTrain;

%% b) Standard deviations/Cov matrix for other classifiers

switch classifier
 case {'nbayes'}
  % all done
  
 case {'nbayesPooled','nbayesPooledResampling'}
  % estimate standard deviations over the entire data, now that
  % they have a common mean of 0 in each feature
  stds = repmat( std(training_set,0,1), nClasses,1 );

 case {'lda','ldaSVD','ldaCV'}
  % estimate a full covariance matrix for the centred training set
  if isempty(params_classifier) useRobustEstimate = 0; else
  useRobustEstimate = params_classifier{1}; end
  fprintf('apply_classifier: useRobustEstimate = %d\n',useRobustEstimate);
  
  if ~useRobustEstimate
    stds = cov(training_set);
  else
    % use estimate from LIBRA
    rew  = mcdcov(training_set,'plots',0);
    stds = rew.cov; 
  end    
    
 case {'qda','qdaSVD','qdaCV'}
  % estimate one covariance matrix per class
  if isempty(params_classifier) useRobustEstimate = 0; else
  useRobustEstimate = params_classifier{1}; end
  fprintf('apply_classifier: useRobustEstimate = %d\n',useRobustEstimate);
  
  classCovarianceMatrices = cell(nClasses,1);
  for c = 1:nClasses

    if ~useRobustEstimate      
      classCovarianceMatrices{c} = cov(training_set(indices{c},:));
    else
      rew = mcdcov(training_set(indices{c},:),'plots',0);
      classCovarianceMatrices{c} = rew.cov; 
    end
  end
end
  

%% c) pack the results into a cell array



% Generative model - each cell contains a cell array
% where each cell has one parameter - mean, covariance matrix, etc
for c=1:1:nClasses
  models{c} = cell(2,1);
  models{c}{1} = means(c,:);
  switch classifier
   case {'nbayes','nbayesPooled','nbayes-unitvariance','nbayesPooledResampling'}
    models{c}{2} = stds(c,:);    
   otherwise
    models{c}{2} = [];
  end
  models{c}{3} = classPriors(c);
end

% Discriminative model - a cell array of sets of weights
models{nClasses+1} = cell(1,1);
  
% Training Set information
trainingSetInfo.nExamples         = nTrain;
trainingSetInfo.nFeatures         = nFeatures;
trainingSetInfo.nClasses          = nClasses; % same thing as labels
trainingSetInfo.sortedLabelValues = sortedLabelValues;
trainingSetInfo.classPriors       = classPriors;
models{nClasses+2} = trainingSetInfo;

switch classifier
 case {'nbayes','nbayesPooled','nbayes-unitvariance'}
 case {'nbayesPooledResampling'}
 case {'lda'}
  models{nClasses+3}{1} = stds; % covariance matrix
 case {'qda'}
  models{nClasses+3}{1} = classCovarianceMatrices;
 case {'ldaSVD','ldaCV'}
  models{nClasses+3}{1} = stds; % covariance matrix
  if isequal(classifier,'ldaSVD')
    models{nClasses+3}{2} = V; % projection into SVD space
  else
    models{nClasses+3}{2}{1} = Vsvd; % project into SVD space
    models{nClasses+3}{2}{2} = Vcv;  % project into CV space of that
  end
  
 case {'qdaSVD','qdaCV'}
  models{nClasses+3}{1} = classCovarianceMatrices; % covariance matrix
  if isequal(classifier,'qdaSVD')
    models{nClasses+3}{2} = V; % projection into SVD space
  else
    models{nClasses+3}{2}{1} = Vsvd; % project into SVD space
    models{nClasses+3}{2}{2} = Vcv;  % project into CV space of that
  end
end
	
%% reduce dimensionality of the training set

function [MM,V] = Principal_Component_Analaysis(training_set,training_set_labels,method,fraction)

switch method

  case {'SVD'}
   % SVD dataset
   [U,S,V] = compute_fastSVD(training_set);
   [nExamples,nVoxels] = size(training_set);  
   maxp = min(nExamples,nVoxels);
   r = rank(S);
   
   % reduce the three matrices
   U = U(:,1:maxp);
   if nExamples < nVoxels; S = S(:,1:maxp); else; S = S(1:maxp,:); end 
   V = V(:,1:maxp);
   
   % scree plot intrinsic dimensionality
   scores           = diag(S);
   normalizedScores = scores.^2;
   normalizedScores = normalizedScores ./ sum(normalizedScores);
   components       = V';
   MM         = U*S;
   
   % find those components - sort and invert order (larger first)
   [sns,sindices] = sort(normalizedScores);
   sns      = flipud(sns);
   sindices = flipud(sindices);
   
   % Swap components into the appropriate order
   components = components(sindices,:);
   scores     = scores(sindices);
   normalizedScores = normalizedScores(sindices);  
   MM         = MM(:,sindices);

   % find the components to keep in order to account for <fraction> of
   % the variance
   tmp = cumsum(sns);
   pos = find(tmp >= fraction);
   nComponentsToKeep = pos(1); % minimum number of components;
   nComponentsToKeep = min(nComponentsToKeep,r);
   
   sindices = 1:nComponentsToKeep;
   
   % crop the matrices accordingly
   components = components(sindices,:);
   scores     = scores(sindices);
   normalizedScores = normalizedScores(sindices);  
   MM         = MM(:,sindices);
   V = components';  
   
   nComponents = length(sindices);
  
 case {'CV'}
  
  training_set_labelsOneOfN = OneOfNencoding(training_set_labels);
  [cvals,V] = canvar( training_set, training_set_labelsOneOfN );
  MM = training_set * V;
  
end

%% Transform a list of labels into a "1 of N" encoding
%% (a binary matrix with as many rows as examples, and as many
%% columns as labels.

function [labels1ofN] = OneOfNencoding(labels)

classes   = unique(labels); nClasses = length(classes);
nExamples = length(labels);

labels1ofN = zeros(nExamples,nClasses);
for c = 1:nClasses
  label           = classes(c);
  labels1ofN(:,c) = (labels == label);
end

function test_2D()

nPerClass = 100;
nTotal    = 2*nPerClass;
trainRange = [1:nTotal/4,nTotal/2+1:3*nTotal/4];
testRange  = [nTotal/4+1:nTotal/2,3*nTotal/4+1:nTotal];
labels     = [repmat(1,nPerClass,1);repmat(2,nPerClass,1)];
labelsTrain   = labels(trainRange);
labelsTest    = labels(testRange);

sigma = [[1 0];[0 1]];
mu1   = [0 0];
mu2   = [2 0];
a = mvnrnd(mu1,sigma,nPerClass);
b = mvnrnd(mu2,sigma,nPerClass);
examples   = [a;b];
examplesTrain = examples(trainRange,:);
examplesTest  = examples(testRange,:);

% means are close 
mu1   = [0 0];
mu2   = [0.1 0];
a = mvnrnd(mu1,sigma,nPerClass);
b = mvnrnd(mu2,sigma,nPerClass);
examples   = [a;b];
examplesTrain = examples(trainRange,:);
examplesTest  = examples(testRange,:);

sigma = [[1 1];[1 1]];

mu1   = [0 0];
mu2   = [2 0];
a = mvnrnd(mu1,sigma,nPerClass);
b = mvnrnd(mu2,sigma,nPerClass);
examples   = [a;b];
examplesTrain = examples(trainRange,:);
examplesTest  = examples(testRange,:);

classifier = 'nbayesPooledResampling';
[models] = classifierNBayes_v2(examplesTrain,labelsTrain,classifier);
[scores] = applyClassifier_v2(examplesTest,models,classifier);
[result1,learntLabels] = summarizePredictions_v2(examplesTest,models,scores,classifier,'accuracy',labelsTest);

classifier = 'nbayesPooled';
[models] = classifierNBayes_v2(examplesTrain,labelsTrain,classifier);
[scores] = applyClassifier_v2(examplesTest,models,classifier);
[result2,learntLabels] = summarizePredictions_v2(examplesTest,models,scores,classifier,'accuracy',labelsTest);

[models] = classifierNBayes_v2(examplesTrain,labelsTrain,'lda');
[scores] = applyClassifier_v2(examplesTest,models,'lda');
[result,learntLabels] = summarizePredictions_v2(examplesTest,models,scores,'nbayes','accuracy',labelsTest);

[models] = classifierNBayes_v2(examplesTrain,labelsTrain,'ldaSVD');
[scores] = applyClassifier_v2(examplesTest,models,'ldaSVD');
[result,learntLabels] = summarizePredictions_v2(examplesTest,models,scores,'nbayes','accuracy',labelsTest);

models{1}{1}
models{1}{2}
models{2}{1}
models{2}{2}

clf;
hold on;
plot(a(:,1),a(:,2),'b.','MarkerSize',6);
plot(b(:,1),b(:,2),'r.','MarkerSize',6);
hold off;

function test_simple()

  means{1} = 1:1:10;
  means{2} = 2 + means{1};
  examples = zeros(40,10);
  labels   = zeros(40,1);
  
  for i=1:1:40
    if mod(i,2)
      examples(i,:) = randn(1,10) + means{1};
      labels(i) = 2;
    else
      examples(i,:) = randn(1,10) + means{2};
      labels(i) = 1;
    end
  end
  
  expInfo.experiment = 'simple';
  expInfo.meta       = [];
  
  training_set    = examples(1:30,:);
  testSet     = examples(31:40,:);
  training_set_labels = labels(1:30);
  testLabels  = labels(31:40);
  
  [models] = classifierLDA(training_set, training_set_labels);
  [scores] = applyClassifier_v2(testSet,models,'nbayes');
  [result,learntLabels] = summarizePredictions(testSet,models,scores,'nbayes','accuracy',testLabels);
  result{1}
  
  % check models{2} {1} and {2} for the mean and variance of class 0
  
function test_unbalanced()

  means{1} = 1:1:10;
  means{2} = 2 + means{1};
  examples = zeros(84,10);
  labels   = zeros(84,1)
  
  % train set
  for i=1:1:50
    examples(i,:) = randn(1,10) + means{1};
    labels(i) = 2;
  end
  
  for i=51:1:60
    examples(i,:) = randn(1,10) + means{2};
    labels(i) = 1;
  end
    
  % test set
  for i=61:1:80
    examples(i,:) = randn(1,10) + means{1};
    labels(i) = 2;
  end
  
  for i=81:1:84
    examples(i,:) = randn(1,10) + means{2};
    labels(i) = 1;
  end

  % pick set
  training_set    = examples(1:60,:);
  testSet     = examples(61:84,:);
  training_set_labels = labels(1:60,:);
  testLabels  = labels(61:84,:);

  [models] = classifierNBayes(training_set, training_set_labels);
  [scores] = applyClassifier(testSet,models,'nbayes');
  [result,learntLabels] = summarizePredictions(testSet,models,scores,'nbayes','accuracy',testLabels);
  result{1}

function test_multiple()

  means{1} = 1:1:10;
  means{2} = 2 + means{1};
  means{3} = 4 + means{1};
  means{4} = 6 + means{1};
  means{5} = 8 + means{1};
  means{6} = 10 + means{1};
  examples = zeros(6*(10+2),10);
  labels = zeros(6*(10+2),1);

  expInfo.experiment = 'simple';
  expInfo.meta       = [];
  
  idx = 1;
  for i=1:1:10
    for c=1:1:6
      examples(idx,:) = randn(1,10) + means{c};
      labels(idx) = c;
      idx = idx + 1;
    end
  end
  nTrain = 10*6;
  
  for i=1:1:2
    for c=1:1:6
      examples(idx,:) = randn(1,10) + means{c};
      labels(idx) = c;
      idx = idx + 1;
    end
  end
  nTest = 2*6;
  
  training_set    = examples(1:nTrain,:);
  testSet     = examples((nTrain+1):(nTrain+nTest),:);
  training_set_labels = labels(1:nTrain,:);
  testLabels  = labels((nTrain+1):(nTrain+nTest),:);
  
  [models] = classifierNBayes(training_set, training_set_labels);
  [scores] = applyClassifier(testSet,models,'nbayes');  
  [result,learntLabels] = summarizePredictions(testSet,models,scores,'nbayes','accuracy',testLabels);
  result{1}

  [models,scores] = trainClassifierL1O(examples,labels,'nbayes','full');
  [result,learntLabels] = summarizePredictions(examples,models,scores,'nbayes','accuracy',labels);

  [models1,scores1] = trainClassifier_kFoldCV(examples,labels,expInfo,'nbayes','full',72);
  [result1,learntLabels1] = summarizePredictions(examples,models1,scores1,'nbayes','accuracy',labels);
