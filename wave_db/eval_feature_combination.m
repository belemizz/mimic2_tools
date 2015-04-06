icd9_code = '428.0';
signal_duration = 10800;

extract_feature_flag = false;

id_path = sprintf('../data/%s.csv', icd9_code);
csvdata = csvread(id_path);

id_list = csvdata(1,:);
mortality_flag = csvdata(2,:);

feature_path = sprintf('../data/f_%s_%d.csv', icd9_code, signal_duration);
metric_list = {'HR', 'RESP', 'SpO2'};

%% preprocesses
if extract_feature_flag
  extract_feature_of(id_list, feature_path, metric_list,signal_duration);
end

%% classification with single metric
n_metrics = length(metric_list);
n_feature_per_metric = 3;
n_feature_for_classification = 2;
comb = combnk(1:n_feature_per_metric, n_feature_for_classification);

loss_result_by_metric = cell(1, n_metrics);
condition_by_metric = cell(1, n_metrics);

for m_idx = 1:n_metrics
  
  loss_array = zeros(1, size(comb,1) );
  condition_array = cell(1, size(comb,1));
  
  for c_idx = 1:size(comb,1)
    f_idx = comb(c_idx,:) + (m_idx-1) * n_feature_per_metric;
    display(m_idx);
    display(f_idx);
    
    loss = classify_feature(feature_path, mortality_flag, f_idx);
    display(loss);
    
    loss_array(c_idx) = loss;
    condition_array{c_idx} = f_idx;
  end
  
  loss_result_by_metric{m_idx} = loss_array;
  condition_by_metric{m_idx} = condition_array;

end

%% classification with variations
loss_result_by_feature = cell(1,3);
condition_by_feature = cell(1,3);

for idx = 1:3
  variation_idx = idx:3:9;
  comb = combnk(variation_idx, n_feature_for_classification);
  loss_array = zeros(1, size(comb,1));
  condition_array = cell(1, size(comb,1));

  for c_idx = 1:size(comb,1);
    f_idx = comb(c_idx,:);
    display(f_idx);
    loss = classify_feature(feature_path, mortality_flag, f_idx);
    loss_array(c_idx) = loss;
    condition_array{c_idx} = f_idx;
  end
  
  loss_result_by_feature{idx} = loss_array;
  condition_by_feature{idx} = condition_array;

end
