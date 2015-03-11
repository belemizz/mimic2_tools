%% add path for toolbox
addpath('../../Matlab/Physionet/Toolbox/wfdb-app-toolbox-0-9-9/mcode');

base = 'mimic2wdb/matched';
id = 's00292';
file = 's00292-3050-10-10-19-46n';
sig_url = sprintf('%s/%s/%s',base, id, file);

metric_no = 1;


%% get the basic infomation of the data
siginfo = wfdbdesc(sig_url);

%% get the waveform data
[tm,sig,a] = rdsamp(sig_url,[],siginfo(metric_no).LengthSamples);

%% figure
figure;
plot(tm/60, sig(:,metric_no));

xlabel('time(min)');
ylabel(siginfo(metric_no).Description);
xlim([0,max(tm)/60]);

yceil = ceil(max(sig(:,metric_no)) / 10) * 10;
ylim([0,yceil]);

