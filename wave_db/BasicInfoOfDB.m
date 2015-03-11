%% add path for toolbox
addpath('../../Matlab/Physionet/Toolbox/wfdb-app-toolbox-0-9-9/mcode');

base = 'mimic2wdb/matched';

list_id = fopen('../clinical_db/numerics_list.dat');
numeric_list = textscan(list_id,'%s');
fclose(list_id);

for idx = 1:10
    file = numeric_list{1}{idx};

    sig_url = sprintf('%s/%s',base, file);
    metric_no = 1;

    %% get id
    pat_id = file(1:6);
    num_id = file(strfind(file, '/') + 1 : length(file));

    %% get the basic infomation of the data
    siginfo = wfdbdesc(sig_url);
    sigdesc = siginfo(metric_no).Description;
    siggain = siginfo(metric_no).Gain;
    sigunit = siggain(strfind(siggain,'/')+1:length(siggain));

    %% filename
    figname = sprintf('%s-%s.png', sigdesc, num_id);

    if siginfo(metric_no).LengthSamples > 1
        %% get the waveform data
        [tm,sig,a] = rdsamp(sig_url,[],siginfo(metric_no).LengthSamples);

        %% figure
        h = figure;
        plot(tm/60, sig(:,metric_no));

        title(num_id);

        xlabel('time(min)');
        ylabel(sprintf('%s [%s]', sigdesc, sigunit));
        xlim([0,max(tm)/60]);

        yceil = ceil(max(sig(:,metric_no)) / 10) * 10;
        ylim([0,yceil]);
        % saveas(h, figname);
    end
end
