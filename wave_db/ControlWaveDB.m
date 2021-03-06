
function ControlWaveDB(mode)
% this function is no longer used
% use draw_graph_of

%% add path for toolbox
addpath('../../Matlab/Physionet/Toolbox/wfdb-app-toolbox-0-9-9/mcode');

base = 'mimic2wdb/matched';
data_folder = '../data';

save_graph = true;
metric_list = {'HR', 'SpO2', 'RESP', 'NBPMean', 'ABPMean'};
%metric_list = {'NBPSys','NBPDias','NBPMean'};
% supported metrics: 'HR', 'PULSE', 'RESP', 'SpO2'

pidx_list = 1:100; % Max:2808
n_pid_per_page = 1;

%% read lists
numerics_all = load_numerics_all();
pid_all = load_pid_all();

% anonymous functions
sig_url = @(nidx) sprintf('%s/%s',base, numerics_all{nidx});
data_path = @(filename) sprintf('%s/%s',data_folder, filename);
get_unit = @(info) info.Gain(strfind(info.Gain,'/')+1:length(info.Gain));
get_start_date = @(info) datetime(strcat(info.StartTime(15:24),',',info.StartTime(2:9)), 'InputFormat', 'dd/MM/yyyy,HH:mm:ss');

% for drawing graphs
switch mode
  case 1
    draw_graphs();
  case 2
    graph_of_icu_expire_flg_eq(true, pidx_list);
  otherwise
    desc_list = list_wave_metric();
    display(desc_list);
    display(length(desc_list));
end
  
%% load Functions
  function numerics_all = load_numerics_all()
    % load the names of the numerics files from the file
    f = fopen('../data/numerics_list.dat');
    temp = textscan(f,'%s');
    numerics_all = temp{1};
    fclose(f);
  end

  function pid_all = load_pid_all()
    f = fopen('../data/id_list.dat');
    pid_all = cell2mat(textscan(f,'%d'));
    fclose(f);
  end

%% admission statics
  function graph_of_icu_expire_flg_eq(flg, pidx)
    all_list = PidOfIcustayExpireFlagEq(flg);
    pid_list = all_list(pidx);
    
    n_all_page = ceil(length(pid_list)/n_pid_per_page);
    for page_idx = 1:n_all_page
      pid_for_page = pid_list(n_pid_per_page * (page_idx-1)+1: min(length(pidx_list),n_pid_per_page * page_idx));
      draw_connected_graph(pid_for_page);
    end
  end

%% graph functions
  function draw_graphs()
    n_all_page = ceil(length(pidx_list)/n_pid_per_page);
    for page_idx = 1:n_all_page
      pid_for_page = pid_all(pidx_list(n_pid_per_page*(page_idx-1)+1 : min(length(pidx_list),n_pid_per_page * page_idx)));
      draw_connected_graph(pid_for_page);
    end
  end

  function draw_connected_graph(pid_list)
    
    % prepare figure
    h = figure;
    
    for pidx = 1:length(pid_list)
      pid = pid_list(pidx);
      % pick numerics
      nidx_list = get_nidx_list_for(pid);
      display(numerics_all(nidx_list));
      
      has_info = false(length(metric_list),1); %flags to check if we get info
      base_time = datetime(zeros(length(metric_list),6));
      unit = cell(length(metric_list),1);
      max_tm = 0;
      
      for nidx = 1:length(nidx_list)
        % get the basic infomation of the data
        info = get_sig_info_of(sig_url(nidx_list(nidx)), metric_list);

        if ~isempty(info)
          signal_length = max([info.LengthSamples]);
          limit_length = 200000;
          split_num = ceil(signal_length / limit_length);
          
          for idx = 1:split_num
            sample_stop = min(limit_length * idx, signal_length);
            sample_start = limit_length*(idx - 1) +1;
            display(sprintf('%s: %d - %d',sig_url(nidx_list(nidx)), sample_start, sample_stop));
            [tm,sig,~] = rdsamp(sig_url(nidx_list(nidx)),[],sample_stop, sample_start);
            
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
%                plot(tm_from_base/60/60, sig(:,info(didx).SignalIndex+1));
                plot(tm_from_base/60/60, sig(:,info(didx).SignalIndex+1),'Color','b');

                max_tm = max(max_tm, max(tm_from_base)/60/60);
                hold on;
              end
            end
          end
        end
      end
      % add axis descripton
      for didx= 1:length(metric_list);
        if has_info(didx)
          subplot(length(metric_list) * n_pid_per_page, 1, length(metric_list) * (pidx-1) + didx);
          title(sprintf('ID:%d   [%s]', pid, datestr(base_time(didx))));
          xlabel('time(hour)');
          ylabel(sprintf('%s\n[%s]', metric_list{didx}, unit{didx}));
          xlim([0,max_tm]);
          ylim([0,inf]);
        end
      end
    end

    if save_graph
      % save graph as a picture
      figname = sprintf('%s-%s.png', strjoin(metric_list,'_'), mat2str(pid_list));
      set(gcf,'PaperUnits','inches','PaperPosition',[0 0 6 1.5*length(pid_list)*length(metric_list)]);
      saveas(h, data_path(figname));
    end
  end

%% description list
  function [metric_list,freq] = list_wave_metric()
    metric_list = {};
    freq = [];
    
    for pidx = pidx_list
      pid = pid_all(pidx);
      nidx_list = get_nidx_list_for(pid);
      
      for nidx = nidx_list
        available_list = get_metric_list(sig_url(nidx));
        for aidx = 1:length(available_list)
          midx = find(ismember(metric_list,available_list{aidx}));
          if midx > 0
            freq(midx) = freq(midx) + 1;
          else
            metric_list = [metric_list, available_list{aidx}];
            freq(length(freq)+1) = 1;
          end
        end
      end
      display(metric_list);
      display(freq);
      display(length(metric_list));
      display(pidx);
    end
  end


%% common functions
  function sig_info = get_sig_info_of(sig_url, metric_list)
    % return signal information for a metric
    info = wfdbdesc(sig_url);
    
    field = fieldnames(info);
    n_field = length(field);
    init = cell(1, n_field);
    
    blank_struct = cell2struct(init, field,2);
    
    if ~isempty(info)
      for midx = 1:length(metric_list)
        metric_in_info = strrep({info.Description}, ' ','');
        metric_index = find(ismember(metric_in_info, metric_list{midx}));
        if metric_index > 0
          sig_info(midx) = info(metric_index);
        else
          sig_info(midx) = blank_struct;
        end
      end
    end
    
    if ~exist('sig_info','var')
      sig_info = struct([]);
    end
  end

  function metric_list = get_metric_list(sig_url)
    info = wfdbdesc(sig_url);
    if length(info)>1
      metric_list = strrep({info.Description}, ' ','');
    else
      metric_list = {};
    end
    
  end

  % return indexs of numerics list for a specific id
  function nidx_list = get_nidx_list_for(pid)
    nidx_list = [];
    for nidx = 1:length(numerics_all)
      if id_of(numerics_all{nidx}) == pid
        nidx_list = [nidx_list, nidx]; %#ok<AGROW>
      end
    end
  end

  function pat_id = id_of(numerics)
    pat_id = str2num(numerics(2:6));
  end
end

