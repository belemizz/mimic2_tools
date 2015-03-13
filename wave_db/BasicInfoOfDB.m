%% add path for toolbox

function BasicInfoOfDB()
addpath('../../Matlab/Physionet/Toolbox/wfdb-app-toolbox-0-9-9/mcode');

base = 'mimic2wdb/matched';
save_graph = true;
metric_no = 1;
%metric_desc = 'HR'

pidx_list = 6:5:30; % Max:2808
n_graph_per_page = 4;

connected_graph = true;

%% read lists
f = fopen('../data/numerics_list.dat');
temp = textscan(f,'%s');
numerics_all = temp{1};
fclose(f);

f = fopen('../data/id_list.dat');
pid_all = cell2mat(textscan(f,'%d'));
fclose(f);

% f = fopen('../data/icu_admission_details.csv');
% admission_info = textscan(f, '%d %d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s', 'Delimiter',',');
% fclose(f);

% anonymous functions
sig_url = @(nidx) sprintf('%s/%s',base, numerics_all{nidx});

draw_graphs();

%desc_list = list_wave_desc();
%display(desc_list);
%display(length(desc_list));


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

  function draw_graphs()
    n_all_page = ceil(length(pidx_list)/n_graph_per_page);
    for page_idx = 1:n_all_page
      patient_id_list = pid_all(pidx_list(n_graph_per_page*(page_idx-1)+1 : min(length(pidx_list),n_graph_per_page * page_idx)));
      if connected_graph
        draw_connected_graph(patient_id_list);
      else
         draw_graph(patient_id_list);
      end
    end
  end

  function draw_graph(pid_list)
    for pidx = 1:length(pid_list)
      pid = pid_list(pidx);
      
      % pick numerics
      nidx_list = get_nidx_list_for(pid);
      display(numerics_all(nidx_list));
      
      % prepare figure
      h = figure('Position',[100 100 300*length(nidx_list) 400]);
      for nidx = 1:length(nidx_list)
        [sig_desc, sig_unit, sig_length, sig_start] = get_signal_info(sig_url(nidx_list(nidx)));

        if sig_length > 1

          % get the waveform data
          [tm,sig,~] = rdsamp(sig_url(nidx_list(nidx)),[],sig_length);

          % figure
          subplot(1, length(nidx_list),nidx);
          plot(tm/60/60, sig(:,metric_no));

          title(sprintf('ID:%d   [%s]', pid, datestr(sig_start)));
          xlabel('time(hour)');
          xlim([0,inf]);
          ylabel(sprintf('%s [%s]', sig_desc, sig_unit));
          ylim([0,inf]);

          if save_graph
            figname = sprintf('%s-%d.png', sig_desc, pid);
            set(gcf,'PaperUnits','inches','PaperPosition',[0 0 3*length(nidx_list) 1.5]);
            saveas(h, figname);
          end
        else
          display(sprintf('%s-%d: no enough data for graph', sig_desc, pid))
        end

      end
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

      subplot(n_graph_per_page, 1, pidx);
      hold on;

      for nidx = 1:length(nidx_list)
        % get the basic infomation of the data
        [sig_desc, sig_unit, sig_length, sig_start] = get_signal_info(sig_url(nidx_list(nidx)));
        if nidx == 1
          base_time = sig_start;
        end
        
        if sig_length > 1

          % get the waveform data
          [tm,sig,~] = rdsamp(sig_url(nidx_list(nidx)),[],sig_length);

          if nidx > 1
            tm = tm + seconds(sig_start - base_time);
          end

          plot(tm/60/60, sig(:,metric_no),'Color', 'b');
        end
      end

      % set title and axis
      title(sprintf('ID:%d   [%s]', pid, datestr(base_time)));
      xlabel('time(hour)');
      ylabel(sprintf('%s [%s]', sig_desc, sig_unit));
      xlim([0,inf]);
      ylim([0,inf]);

    end
    
    if save_graph
      % save graph as a picture
      figname = sprintf('%s-%s.png', sig_desc, mat2str(pid_list));
      set(gcf,'PaperUnits','inches','PaperPosition',[0 0 6 1.5*length(pid_list)]);
      saveas(h, figname);
    end
  end

  % return signal information for drawing graphs
  function [sig_desc, sig_unit, sig_length, sig_start] = get_signal_info(sig_url)
    siginfo = wfdbdesc(sig_url);

    sig_desc = siginfo(metric_no).Description;

    siggain = siginfo(metric_no).Gain;
    sig_unit = siggain(strfind(siggain,'/')+1:length(siggain));
    
    sig_length = siginfo(metric_no).LengthSamples;
    
    sig_start = siginfo(metric_no).StartTime;
    start_date = sig_start(15:24);
    start_time = sig_start(2:9);
    sig_start = datetime(strcat(start_date,',',start_time), 'InputFormat', 'dd/MM/yyyy,HH:mm:ss');
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

