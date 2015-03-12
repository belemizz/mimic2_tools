%% add path for toolbox

function BasicInfoOfDB()
addpath('../../Matlab/Physionet/Toolbox/wfdb-app-toolbox-0-9-9/mcode');

base = 'mimic2wdb/matched';
metric_no = 1;
n_graph = 5;
save_graph = true;

%% read lists
f = fopen('../data/numerics_list.dat');
temp = textscan(f,'%s');
numerics_list = temp{1};
fclose(f);

f = fopen('../data/id_list.dat');
id_list = cell2mat(textscan(f,'%d'));
fclose(f);

f = fopen('../data/icu_admission_details.csv');
admission_info = textscan(f, '%d %d %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s', 'Delimiter',',');
fclose(f);

%% generate patient id list from admission list
for page_idx = 4:4
  patient_id_list = id_list(page_idx*(n_graph-1)+1:page_idx*n_graph);
%  draw_connected_graph(patient_id_list);

  draw_graph(patient_id_list);
  
end

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

  function draw_connected_graph(pid_list)
    
    % prepare figure
    h = figure;
    
    for pidx = 1:length(pid_list)
      pid = pid_list(pidx);
      % pick numerics
      nidx = get_nidx_for(pid);
      display(numerics_list(nidx));

      subplot(length(pid_list), 1, pidx);
      hold on;

      for index = 1:length(nidx)
        sig_url = sprintf('%s/%s',base, numerics_list{nidx(index)});

        % get the basic infomation of the data
        [sig_desc, sig_unit, sig_length, sig_start] = get_signal_info(sig_url);

        if index == 1
          base_time = sig_start;
        end
        
        if sig_length > 1

          % get the waveform data
          [tm,sig,~] = rdsamp(sig_url,[],sig_length);

          if index > 1
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
      

  function draw_graph(pid_list)
    for pidx = 1:length(pid_list)
      pid = pid_list(pidx);
      
      % pick numerics
      nidx = get_nidx_for(pid);
      display(numerics_list(nidx));
      
      % prepare figure
      h = figure('Position',[100 100 300*length(nidx) 400]);
      for index = 1:length(nidx)
        sig_url = sprintf('%s/%s',base, numerics_list{nidx(index)});
        [sig_desc, sig_unit, sig_length, sig_start] = get_signal_info(sig_url);
        

        if sig_length > 1

          % get the waveform data
          [tm,sig,~] = rdsamp(sig_url,[],sig_length);

          % figure
          subplot(1, length(nidx),index);
          plot(tm/60/60, sig(:,metric_no));

          title(sprintf('ID:%d   [%s]', pid, datestr(sig_start)));
          xlabel('time(hour)');
          xlim([0,inf]);
          ylabel(sprintf('%s [%s]', sig_desc, sig_unit));
          ylim([0,inf]);

          if save_graph
            figname = sprintf('%s-%d.png', sig_desc, pid);
            set(gcf,'PaperUnits','inches','PaperPosition',[0 0 3*length(nidx) 1.5]);
            saveas(h, figname);
          end
        else
          display(sprintf('%s: no enough data for graph', numerics_for_id{index}))
        end

      end
    end
  end

  % return indexs of numerics list for a specific id
  function nidx = get_nidx_for(id)
    nidx = [];
    for index = 1:length(numerics_list)
      if id_of(numerics_list{index}) == id
        nidx = [nidx, index];
      end
    end
  end

  function pat_id = id_of(numerics)
    pat_id = str2num(numerics(2:6));
  end


end

