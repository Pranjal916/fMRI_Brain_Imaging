% kNN_Classifier
%
% Train a neural network classifier using cross validation within
% the training set to decide when to stop.
%
% Example:
%  - classifier=kNN_Classifier(examples,labels,{20,50,50,'scg'})

function [net] = kNN_Classifier( varargin )

DEBUG = 0;

l = length(varargin);
if l < 1; help kNN_Classifier; return; end

examples = varargin{1}; [nExamples,nFeatures] = size(examples);
labels   = varargin{2}; sortedLabels = unique(labels); nLabels = length(sortedLabels);

params_classifier = {};
if l > 2
  params_classifier = varargin{3};
end

nIn     = nFeatures;
nHidden = nLabels;
nOut    = nLabels;
outfunc = 'softmax'; % good setting for classification problems
optimiz = 'scg'; % scaled conjugate gradient
iterParams = [];
nIter   = 0; % if 0 -> use CV within the trainining set to find # iterations

k = length(params_classifier);
if k > 0
  nHidden = params_classifier{1};
  if k > 1
    nIter = params_classifier{2};
    if k > 2
      iterParams = params_classifier{3};
      if k > 3
	optimiz = params_classifier{4};
      end
    end
  end
end

%
% Train it
% 

examples = normalize(examples);
labelsN  = multipleEncoding(labels);

if nHidden
  % train a MLP
  net = mlp( nIn, nHidden, nOut, outfunc );
  % HACK to pass the network type with the network
  net.ourNetType = 'MLP';
else
  % if nHidden = 0, train a Generalised Linear Model
  net = glm( nIn, nOut, outfunc );
  net.ourNetType = 'GLM';
end
fprintf('training a %s neural network\n',net.ourNetType);

if nIter > 0
  % number of iterations has been specified    
else
  % determine nIter via cross validation within the training set
  nIter = computeOptimalNiter(examples,labels,net,nHidden,nIter,iterParams,DEBUG);
end

options     = zeros(1,18);
options(1)  = 1;                 % Print out error values    
options(14) = nIter;  

if 1
  net = netopt( net, options, examples, labelsN, optimiz);
else
  if nHidden
    net = netopt( net, options, examples, labelsN, optimiz);
  else
    net = glmtrain( net, options, examples, labelsN );
  end
end

%% Normalize each feature to have mean 0 and standard deviation 1

function [Y] = normalize(X)

[nExamples,nFeatures] = size(X);
meanX = mean(X,1);
stdvX = std(X,0,1);

Y = X -  repmat(meanX,[nExamples,1]);
Y = Y ./ repmat(stdvX,[nExamples,1]);

%
% Cross validation within training set to determine how many
% iterations to train the network for
%

function [nIter] = computeOptimalNiter( varargin )

l = length(varargin);
examples = varargin{1};
labels   = varargin{2};
net      = varargin{3};
nHidden  = varargin{4};
nIter    = varargin{5};
iterParams = varargin{6};
if l > 6; DEBUG = varargin{7}; else; DEBUG = 0; end

sortedLabels = unique(labels); nLabels = length(sortedLabels);

for l=1:nLabels
  label   = sortedLabels(l);

  if nIter == 0
    % leave examples out

    places     = find(labels == label);
    nblocks(l) = length(places);
    for p=1:nblocks(l)
      orgSets{l}{p} = [places(p) places(p)];
    end

  elseif nIter == -1
    places  = find(labels == label);
    lastp   = find(diff(places)>1);
    breakp  = lastp + 1;
    breakp  = [1; breakp]; % block beginnings
    lastp   = [lastp; length(places)];
    nblocks(l) = length(breakp);

    % now compute the index intervals in the example array
    % corresponding to each block of examples
    orgSets{l} = cell(nblocks(l),1);
    for p=1:nblocks(l)
      orgSets{l}{p} = [places(breakp(p)) places(lastp(p))];
    end
  else
    fprintf('error: nIter=%d is not supported\n',nIter);pause;return
  end
    
end;


%% Transform a list of labels into a "1 of N" encoding

function [labels1ofN] = multipleEncoding(labels)

classes   = unique(labels); nClasses = length(classes);
nExamples = length(labels);

labels1ofN = zeros(nExamples,nClasses);
for c = 1:nClasses
  label           = classes(c);
  labels1ofN(:,c) = (labels == label);
end



%% check that all labels have the same number of trials
if sum(diff(nblocks))
  fprintf('error: # of trials \n');
  return;
else
  trialsNumber = nblocks(1);
end

if DEBUG
  fprintf('blocks for all\n');
  for l=1:nLabels
    label = sortedLabels(l);
    fprintf('\tlabel %d\t\n',label);
    for p=1:nblocks(l)
      fprintf('\t\t[%d,%d]\n',orgSets{l}{p});
      fprintf('\n');
    end
  end
  fprintf('\n');
  pause
end

if isempty(iterParams)
  nFolds = trialsNumber;
else
  nFolds = iterParams(1);
end

trainImagesPerFold    = cell(nFolds,1);
trainIntervalsPerFold = cell(nFolds,1);
testImagesPerFold     = cell(nFolds,1);
testIntervalsPerFold  = cell(nFolds,1);

for k=1:nFolds
  
  % find the test intervals
  testImagesPerFold{k} = [];
  testIntervalsPerFold{k} = {};
  
  for l=1:nLabels
    ii = orgSets{l}{k}; % image interval
    testImagesPerFold{k} = [testImagesPerFold{k},ii(1):ii(2)];
    testIntervalsPerFold{k}{l} = ii;
  end
  
  % find the train intervals
  trainImagesPerFold{k} = [];
  trainIntervalsPerFold{k} = {};idx = 1;
  
  for ak=1:k-1
    for l=1:nLabels
      ii = orgSets{l}{ak}; % image interval
      trainImagesPerFold{k} = [trainImagesPerFold{k},ii(1):ii(2)];
      trainIntervalsPerFold{k}{idx} = ii; idx = idx + 1;
    end
  end
  
  for ak=k+1:nFolds
    for l=1:nLabels
      ii = orgSets{l}{ak}; % image interval
      trainImagesPerFold{k} = [trainImagesPerFold{k},ii(1):ii(2)];
      trainIntervalsPerFold{k}{idx} = ii; idx = idx + 1;
    end
  end
end;
end
   
%% Run the cross validation

nIterPerFold = zeros(1,nFolds);
errorPerFold = zeros(1,nFolds);
size(labels)
for k=1:nFolds
  fprintf('Testing over multiple fold %d\n',k);
  trainExamples   = examples(trainImagesPerFold{k},:);
  trainLabels     = labels(trainImagesPerFold{k},:);
  trainLabels1ofN = multipleEncoding(trainLabels);
  testExamples    = examples(testImagesPerFold{k},:);
  testLabels      = labels(testImagesPerFold{k},:);
  testLabels1ofN  = multipleEncoding(testLabels);

  % set up training
  nIterPerBurst   = 10;
  nBursts         = 20;
  errorAfterBurst{k} = zeros(1,nBursts);
  
  % train network
  foldNet = net;
  options = zeros(1,18);
  options(1)  = 1;       
  options(14) = nIterPerBurst;
  method = 'scg';
  for b = 1:nBursts
    foldNet = netopt(foldNet, options, trainExamples, trainLabels1ofN, method);
    if nHidden
      yt = mlpfwd(foldNet, testExamples);
    else
      yt = glmfwd(foldNet, testExamples);
    end
      
    [yvalue,ypos]   = max(yt,[],2);
    predictedLabels = sortedLabels(ypos);
    errorAfterBurst{k}(b) = sum(predictedLabels~=testLabels)/length(testLabels);
  end

  [errorPerFold(k),nBurstsPerFold(k)] = min(errorAfterBurst{k});
end

nIter = ceil(median(nBurstsPerFold))*nIterPerBurst;

