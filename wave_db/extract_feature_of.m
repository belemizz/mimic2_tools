function extract_feature_of( id_list, algorithm, output_path)
% extract feature of ids in id_list with alogrithm

if nargin < 3
  output_path = '../data/feature_output.csv';
end

set_path;

numerics_all = load_numerics_all();
feature_list = [];
for pidx = 1:length(id_list)
  pid = id_list(pidx);
  nurl_list = get_nurl_list_for(pid, numerics_all);
  
  if ~isempty(nurl_list)
    sig_url = nurl_list{length(nurl_list)};
    display(sig_url);
    feature = extract_feature_from(sig_url);
  else
    display(sprintf('No signal info for %d', id_list(pidx)));
    feature = [NaN, NaN];
  end
  
  feature_list = [feature_list;feature];
  
end

display([id_list', feature_list]);
csvwrite(output_path, feature_list);
end

function feature = extract_feature_from(sig_url)
metric_list = {'HR'};
duration = 7200; %sec

info = get_sig_info_of(sig_url, metric_list);

feature = [NaN, NaN];
if ~isempty(info)
  signal = get_signal_index(info,duration);
  [tm,sig,~] = rdsamp(sig_url,[],signal.End, signal.Start);
  hr_signal = sig(:,info(1).SignalIndex+1);
  [~, sig_r, ~] = reliable_signal(tm, hr_signal, 5);
  
  feature = [mean(sig_r) var(sig_r)];
end
end
