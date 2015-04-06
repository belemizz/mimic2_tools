function extract_feature_of( id_list, output_path, metric_list, duration, var_width)
% extract feature of ids in id_list with alogrithm

if nargin < 2
  output_path = '../data/feature_output.csv';
end
if nargin < 3
  metric_list = {'HR','RESP'};
end
if nargin < 4
  duration = 7200; % sec
end
if nargin < 5
  var_width = 7;
end

set_path;

numerics_all = load_numerics_all();
feature_list = [];

feature_per_metric = 3;
feature_dim = feature_per_metric * length(metric_list);

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
    
    coef_feature = linear_reg_feature(tm_r,sig_r, 2);
    
    feature = [feature coef_feature mean(sig_r) var(sig_r)];
  end
end
end

function coef_feature = linear_reg_feature(tm, sig, order)
if nargin < 3
  order = 2; %only 1st order coefficient
end
  
feature_dim = length(order);
coef_feature = NaN(1,feature_dim);

if ~isempty(tm)
  l_reg_model = fitlm(tm, sig); % linear regression
  coefficients = l_reg_model.Coefficients.Estimate';
  coef_feature = coefficients(order);
end
end
