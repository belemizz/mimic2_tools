function extract_feature_of( id_list, algorithm )
% extract feature of ids in id_list with alogrithm

set_path;
data_folder = '../data';

numerics_all = load_numerics_all();

for pidx = 1:length(id_list)
  pid = id_list(pidx);
  nurl_list = get_nurl_list_for(pid, numerics_all);
  
  sig_url = nurl_list{length(nurl_list)};
  display(sig_url);
  
  feature = extract_feature_from(sig_url);

  display(feature);
end

end

function feature = extract_feature_from(sig_url)
  metric_list = {'HR'};
  duration = 3600; %sec
  
  info = get_sig_info_of(sig_url, metric_list);

  feature = NaN;
  if ~isempty(info)
    signal = get_signal_index(info,duration);
    [tm,sig,~] = rdsamp(sig_url,[],signal.End, signal.Start);
    feature = mean(sig(:,info(1).SignalIndex+1));
  end
 end
