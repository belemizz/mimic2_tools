%% add path for toolbox
addpath('../../Matlab/Physionet/Toolbox/wfdb-app-toolbox-0-9-9/mcode');

%% get the basic infomation of the data
siginfo = wfdbdesc('mimic2wdb/matched/s00292/s00292-3050-10-10-19-46n');

%% get the waveform data
[tm,sig,a] = rdsamp('mimic2wdb/matched/s00292/s00292-3050-10-10-19-46n',[],1000);

figure;
plot(tm/60, sig(:,1));
xlabel('time(min)');
ylabel('HR(cycle/sec)');
xlim([0,120]);
ylim([0,200]);

% figure;
% plot(tm/60, sig(:,8));
% xlabel('time(min)');
% ylabel('SpO2(%)');
% xlim([0,120]);
% ylim([0,100]);
% 
