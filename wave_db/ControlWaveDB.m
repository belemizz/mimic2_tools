%% add path for toolbox

function ControlWaveDB(mode)
addpath('../../Matlab/Physionet/Toolbox/wfdb-app-toolbox-0-9-9/mcode');

base = 'mimic2wdb/matched';
data_folder = '../data';

save_graph = true;
metric_list = {'HR', 'RESP'};
% supported metrics: 'HR', 'PULSE', 'RESP', 'SpO2'

pidx_list = 1:1; % Max:2808
n_pid_per_page = 1;

%% read lists
numerics_all = load_numerics_all();
pid_all = load_pid_all();
%admission_info = load_admission_info();

% anonymous functions
sig_url = @(nidx) sprintf('%s/%s',base, numerics_all{nidx});
data_path = @(filename) sprintf('%s/%s',data_folder, filename);
get_unit = @(info) info.Gain(strfind(info.Gain,'/')+1:length(info.Gain));
get_start_date = @(info) datetime(strcat(info.StartTime(15:24),',',info.StartTime(2:9)), 'InputFormat', 'dd/MM/yyyy,HH:mm:ss');

% for drawing graphs
switch mode
  case 1
    draw_graphs();
  otherwise
    desc_list = list_wave_desc();
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

  function admission_info = load_admission_info()
   f = fopen('../data/icu_admission_details.csv');
   admission_info = textscan(f, '%d %d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s', 'Delimiter',',');
   fclose(f);
  end    

%% graph functions
  function draw_graphs()
    n_all_page = ceil(length(pidx_list)/n_pid_per_page);
    for page_idx = 1:n_all_page
      patient_id_list = pid_all(pidx_list(n_pid_per_page*(page_idx-1)+1 : min(length(pidx_list),n_pid_per_page * page_idx)));
      draw_connected_graph(patient_id_list);
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
      
      for nidx = 1:length(nidx_list)
        % get the basic infomation of the data
        info = get_sig_info_of(sig_url(nidx_list(nidx)), metric_list);

        if length(info) > 1
          [tm,sig,~] = rdsamp(sig_url(nidx_list(nidx)),[],max([info.LengthSamples]));        

          for didx= 1:length(metric_list);
            if has_info(didx)
              tm = tm + seconds(get_start_date(info(didx)) - base_time(didx));
            else
              has_info(didx) = true;
              base_time(didx) = get_start_date(info(didx));
              unit{didx} = get_unit(info(didx));
            end
            
            subplot(length(metric_list) * n_pid_per_page, 1, length(metric_list) * (pidx-1) + didx);
            plot(tm/60/60, sig(:,info(didx).SignalIndex+1),'Color', 'b');
            hold on;
          end
          
        end
      end
      % add axis descripton
      for didx= 1:length(metric_list);
        if has_info
          subplot(length(metric_list) * n_pid_per_page, 1, length(metric_list) * (pidx-1) + didx);
          title(sprintf('ID:%d   [%s]', pid, datestr(base_time(didx))));
          xlabel('time(hour)');
          ylabel(sprintf('%s [%s]', metric_list{didx}, unit{didx}));
          xlim([0,inf]);
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
  function desc_list = list_wave_desc()
    desc_list = {};
    for pidx = pidx_list
      pid = pid_all(pidx);
      nidx_list = get_nidx_list_for(pid);
      
      for nidx = nidx_list
        sig_desc_list = get_sig_desc_list(sig_url(nidx));
        desc_list = union(desc_list, sig_desc_list);
      end
      display(desc_list);
      display(length(desc_list));
      display(pidx);
    end
  end


%% common functions
  function sig_info = get_sig_info_of(sig_url, metric_list)
    % return signal information for a metric
    info = wfdbdesc(sig_url);
    
    for midx = 1:length(metric_list)
      metric_index = find(ismember({info.Description}, metric_list{midx}));
      if metric_index > 0
        sig_info(midx) = info(metric_index);
      end
    end
  end

  function sig_desc_list = get_sig_desc_list(sig_url)
    siginfo = wfdbdesc(sig_url);
    sig_desc_list = {siginfo.Description};
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

