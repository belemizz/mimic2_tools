function extract_feature_of( id_list, output_path, metric_list)
% extract feature of ids in id_list with alogrithm

if nargin < 2
  output_path = '../data/feature_output.csv';
end
if nargin < 3
  metric_list = {'HR','RESP'};
end

set_path;

numerics_all = load_numerics_all();
feature_list = [];
feature_dim = 4 * length(metric_list);
var_width = 5;
duration = 10800; %sec

for pidx = 1:length(id_list)
  pid = id_list(pidx);
  nurl_list = get_nurl_list_for(pid, numerics_all);
  
  if ~isempty(nurl_list)
    sig_url = nurl_list{length(nurl_list)};
    display(sig_url);
    feature = extract_feature_from(sig_url, metric_list, duration, var_width);
  else
    display(sprintf('No signal info for %d', id_list(pidx)));
    feature = NaN(1, feature_dim);
  end
  feature_list = [feature_list;feature];
end

display([id_list', feature_list]);
csvwrite(output_path, feature_list);
end

function feature = extract_feature_from(sig_url,metric_list, duration, var_width)

info = get_sig_info_of(sig_url, metric_list);
feature = [];

if ~isempty(info)
  signal = get_signal_index(info,duration);
  [tm,sig,~] = rdsamp(sig_url,[],signal.End, signal.Start);
  for idx = 1:length(metric_list)
    single_signal = sig(:,info(idx).SignalIndex+1);
    [tm_r, sig_r, ~] = reliable_signal(tm, single_signal, var_width);
    
    coef_feature = [NaN NaN];
    if ~isempty(tm_r)
      l_reg_model = fitlm(tm_r, sig_r); % linear regression
      coef_feature = l_reg_model.Coefficients.Estimate';
    end
    feature = [feature coef_feature mean(sig_r) var(sig_r)];
  end
end
end
