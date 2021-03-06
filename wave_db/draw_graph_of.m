function [  ] = draw_graph_of( id_list, metric_list, save_graph, duration, exclude_unreliable)
% draw graph of given ids and metrics
%
% Required Parameters:
%
% id_list
%   array of subject id of the data
%
% metric_list
%   cell array of metrics of interest
%
%
% Optional Parameters:
%
% save_graph
%   if true save the output graph(s)
%
% duration
%   length of signal to be plotted in seconds
%   when inf, then plot all available data
%
% exclude_unreliable
%   if true exclude unreliable datapoints

% default values
if nargin<3
  save_graph = false;
end
if nargin<4
  duration = inf; %draw graph of all value;
end
if nargin<5
  exclude_unreliable = true;
end

set_path;
data_folder = '../data';

n_pid_per_page = max(1, floor(5/length(metric_list)));
n_all_page =ceil(length(id_list) /n_pid_per_page);

%read list
numerics_all = load_numerics_all();

% anonimous functions
get_start_date = @(info) datetime(strcat(info.StartTime(15:24),',',info.StartTime(2:9)), 'InputFormat', 'dd/MM/yyyy,HH:mm:ss');
get_unit = @(info) info.Gain(strfind(info.Gain,'/')+1:length(info.Gain));

% parameters
limit_length = 200000; %length of data that is accessed one time
has_info = false(length(metric_list),1); %flags to check if we get info
base_time = datetime(zeros(length(metric_list),6));
unit = cell(length(metric_list),1);
max_tm = 0;
min_tm = Inf;

for page_idx = 1:n_all_page
  start_pidx = n_pid_per_page*(page_idx-1)+1;
  end_pidx = min(n_pid_per_page * page_idx, length(id_list));
  pid_for_page = id_list(start_pidx:end_pidx);
  draw_page(pid_for_page);
end

  function draw_page(pid_list)
    % draw a page
    
    h = figure;
    
    for pidx = 1:length(pid_list)
      pid = pid_list(pidx);
      % pick numerics url
      nurl_list = get_nurl_list_for(pid, numerics_all);
      
      has_info = false(length(metric_list),1); %flags to check if we get info
      base_time = datetime(zeros(length(metric_list),6));
      unit = cell(length(metric_list),1);
      max_tm = 0;
      min_tm = Inf;
      
      if isinf(duration)
        for nidx = 1:length(nurl_list)
          nurl = nurl_list{nidx};
          plotdata(pidx,nurl);
        end
      else
        if ~isempty(nurl_list)
          plotdata(pidx, nurl_list{length(nurl_list)});
        end
      end
      
      % add axis descripton
      for didx= 1:length(metric_list);
        if has_info(didx)
          subplot(length(metric_list) * n_pid_per_page, 1, length(metric_list) * (pidx-1) + didx);
          title(sprintf('ID:%d   [%s]', pid, datestr(base_time(didx))));
          xlabel('time(hour)');
          ylabel(sprintf('%s\n[%s]', metric_list{didx}, unit{didx}));
          %          xlim([min_tm,max_tm]);
          if isinf(duration)
            xlim([min_tm,max_tm]);
          else
            xlim([max_tm-duration/60/60,max_tm]);
          end
          ylim([0,inf]);
        end
      end
    end
    
    if save_graph
      % save graph as a picture
      data_path = sprintf('%s/%s-%s-%d.png', data_folder, strjoin(metric_list,'_'), mat2str(pid_list), exclude_unreliable);
      set(gcf,'PaperUnits','inches','PaperPosition',[0 0 6 1.5*n_pid_per_page*length(metric_list)]);
      saveas(h, data_path);
    end
  end

  function plotdata(pidx, nurl)
    % plot the data of list of subjects
    % get the basic infomation of the data
    info = get_sig_info_of(nurl, metric_list);
    
    if ~isempty(info)
      signal = get_signal_index(info, duration);
      split_num = ceil( (signal.End - signal.Start + 1) / limit_length);
      
      for idx = 1:split_num
        sample_start = limit_length*(idx - 1) +signal.Start;
        sample_end = min(limit_length * idx + signal.Start-1, signal.End);
        display(sprintf('%s: %d - %d',nurl, sample_start, sample_end));
        [tm,sig,~] = rdsamp(nurl,[],sample_end, sample_start);
        cycle = (tm(length(tm))-tm(1))/(length(tm)-1);
        display(sprintf('cycle: %d',cycle));
        
        for didx= 1:length(metric_list);
          if ~isempty(info(didx).LengthTime)
            
            tm_from_base = tm;
            if has_info(didx)
              tm_from_base = tm + seconds(get_start_date(info(didx)) - base_time(didx));
            else
              has_info(didx) = true;
              base_time(didx) = get_start_date(info(didx));
              unit{didx} = get_unit(info(didx));
            end
            
            sig_of_int = sig(:,info(didx).SignalIndex+1);
            
            subplot(length(metric_list) * n_pid_per_page, 1, length(metric_list) * (pidx-1) + didx);
            hold on;
            
            if exclude_unreliable
              [tm_r, signal_r, tm_e] = reliable_signal(tm_from_base, sig_of_int);
              plot(tm_r/60/60, signal_r, 'b');
              plot(tm_e/60/60, zeros(size(tm_e)), 'r.');
            else
              plot(tm_from_base(~isnan(sig_of_int))/60/60, sig_of_int(~isnan(sig_of_int)),'b');
            end
            
            max_tm = max(max_tm, max(tm_from_base)/60/60);
            min_tm = min(min_tm, min(tm_from_base)/60/60);
          end
        end
      end
    end
  end
end
