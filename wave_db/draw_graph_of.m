function [  ] = draw_graph_of( id_list, metric_list, save_graph, length_of_data)
% draw graph of given ids and metrics
% length_of_data:
%   length of signal to be plotted in seconds
%   when 0, then plot all available data

% default values
if nargin<3
  save_graph = false;
end
if nargin<4
  length_of_data = 0; %draw graph of all value;
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
      min_tm = inf;
      
      if length_of_data > 0
        plotdata(pidx, nurl_list{length(nurl_list)});
      else
        for nidx = 1:length(nurl_list)
          nurl = nurl_list{nidx};
          plotdata(pidx,nurl);
        end
      end

      % add axis descripton
      for didx= 1:length(metric_list);
        if has_info(didx)
          subplot(length(metric_list) * n_pid_per_page, 1, length(metric_list) * (pidx-1) + didx);
          title(sprintf('ID:%d   [%s]', pid, datestr(base_time(didx))));
          xlabel('time(hour)');
          ylabel(sprintf('%s\n[%s]', metric_list{didx}, unit{didx}));
          xlim([min_tm,max_tm]);
          ylim([0,inf]);
        end
      end
    end
    
    if save_graph
      % save graph as a picture
      data_path = sprintf('%s/%s-%s.png', data_folder, strjoin(metric_list,'_'), mat2str(pid_list));
      set(gcf,'PaperUnits','inches','PaperPosition',[0 0 6 1.5*length(pid_list)*length(metric_list)]);
      saveas(h, data_path);
    end
  end

  function plotdata(pidx, nurl)
    % plot the data of list of subjects
    
    % get the basic infomation of the data
    info = get_sig_info_of(nurl, metric_list);
    
    
    if ~isempty(info)
      signal_end = max([info.LengthSamples]);
      signal_freq = max([info.SamplingFrequency]);
      
      if length_of_data > 0
        if signal_freq == 1
          signal_start = max(1, signal_end - length_of_data + 1);
        else
          signal_start = max(1,signal_end - floor(length_of_data / 60) + 1);
        end
      else
        signal_start = 1;
      end

      split_num = ceil( (signal_end - signal_start + 1) / limit_length);
      
      for idx = 1:split_num
        sample_start = limit_length*(idx - 1) +signal_start;
        sample_stop = min(limit_length * idx + signal_start, signal_end);
        display(sprintf('%s: %d - %d',nurl, sample_start, sample_stop));
        [tm,sig,~] = rdsamp(nurl,[],sample_stop, sample_start,1,1);
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
            
            subplot(length(metric_list) * n_pid_per_page, 1, length(metric_list) * (pidx-1) + didx);
            plot(tm_from_base/60/60, sig(:,info(didx).SignalIndex+1),'Color','b');
            
            max_tm = max(max_tm, max(tm_from_base)/60/60);
            min_tm = min(min_tm, min(tm_from_base)/60/60);
            hold on;
          end
        end
      end
    end
  end
end
