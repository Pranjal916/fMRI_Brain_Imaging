% plot_results(info,data,metadata,trial_number,[title_holder],[interactivity=0|1
% Plot the voxel time courses for trial number trial_number, one z slice at a time, for the given IDM. Lays out voxel plots according to their geometric relationship.
% trial_number : number of the trials.
% title_holder : optional text string.
% Example:
%  - plot_results(info,data,metadata,3,'04847',1)

function [] = plot_results( varargin )
  % process arguments
  l = length(varargin);
  if l < 4
    return;
  end

  info  = varargin{1};
  data  = varargin{2};
  metadata  = varargin{3};
  trial_number = varargin{4};

  title_holder   = '';
  interactivity = 1;
  if l > 4
    title_holder = varargin{5};
    if l > 5      
      interactivity = varargin{6};
    end
  end
  clf reset;
  nvoxels      = size(data{1},2);
  colours = {'w'};
  assignments = ones(nvoxels,1);
    
% data dependent parameters
  trialBegin    = 1;
  trialEnd      = info(trial_number).len;
  vmin = sort(min(data{trial_number}(trialBegin:1:trialEnd,:)));
  vmax = sort(max(data{trial_number}(trialBegin:1:trialEnd,:)));
  ntouse = floor(0.05*nvoxels);
  maxActivation = mean(vmax(nvoxels-ntouse+1:1:nvoxels));
  ntouse = floor(0.1*nvoxels);
  minActivation = mean(vmin(1:1:ntouse));
  fprintf(1,'min=%1.2f\tmax=%1.2f\n',minActivation,maxActivation);
  
  trialdata = data{trial_number};
  len       = info(trial_number).len;
  condition = info(trial_number).cond;
  
  % calculate maximum width and height of plot grid
  xMin    = min(metadata.colToCoord(:,1));
  xMax    = max(metadata.colToCoord(:,1));
  columns = 1 + xMax - xMin;
  yMin    = min(metadata.colToCoord(:,2));
  yMax    = max(metadata.colToCoord(:,2));
  rows    = 2 + yMax - yMin;   	% one extra to leave room for grid title
  xOffset = xMin-1;
  yOffset = yMin-1;

  assignments = ones(nvoxels,1);
  slices = unique(metadata.colToCoord(:,3));
  nsubs  = rows * columns;

  for s=1:1:length(slices)
    z = slices(s);
    disp(z)
    sliceVoxels = find(metadata.colToCoord(:,3)==z);
    % for each voxel in plane z, plot its time course at its xy coordinate
    xminPlot=9999; xmaxPlot=-9999; yminPlot=9999; ymaxPlot=-9999;
    for v=1:1:length(sliceVoxels)
      coord = metadata.colToCoord(sliceVoxels(v),:); 
      x=coord(1); y= coord(2);
      xminPlot=min(xminPlot,x); xmaxPlot=max(xmaxPlot,x);
      yminPlot=min(yminPlot,y); ymaxPlot=max(ymaxPlot,y);
      timeCourse = trialdata(:,sliceVoxels(v));
      timeCourse = timeCourse( trialBegin:1:trialEnd,: );
      plotIdx = nsubs - ((y-yOffset-1)*columns + (columns-(x-xOffset)));
      subplot(rows,columns,plotIdx);
      h = plot(timeCourse);	
      set(gca,'XTickLabel',{''});
      set(gca,'YTickLabel',{''});
      axis([0,trialEnd,minActivation+1,maxActivation]);      
    end;
    subplot(rows,columns,(columns/2));
    set(gca,'XTickLabel',{''});
    set(gca,'YTickLabel',{''});
    tstr=sprintf('%s %s subject%s, trial number %d', title_holder,metadata.study,metadata.subject,trial_number);
    tstr=sprintf('%s\nregion %s\n',tstr,metadata.roi);
    tstr=sprintf('%sz=%d x=[%d,%d] ',tstr,z,xminPlot,xmaxPlot);
    tstr=sprintf('%s y=[%d,%d], timeSteps=%d, condition=%d ', tstr,yminPlot,ymaxPlot,trialEnd,condition);
    tstr=sprintf('%s amplitude=[%1.1f %1.1f]', tstr, minActivation+1,maxActivation);
    title(tstr);
  end