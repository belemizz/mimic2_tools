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
  
%  feature_of(nurl_list);
  feature = extract_feature_from(sig_url);

  display(feature);
end
end

function feature = extract_feature_from(sig_url)
  metric_list = {'HR'};
  info = get_sig_info_of(sig_url, metric_list);

  feature = NaN;
  if ~isempty(info)
    signal_length = max([info.LengthSamples]);
    feature = signal_length;
  end
  
 end
