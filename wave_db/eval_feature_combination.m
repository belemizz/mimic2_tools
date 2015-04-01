csvdata = csvread('../data/428.0.csv');
id_list = csvdata(1,:);
mortality_flag = csvdata(2,:);

extract_feature_flag = false;

metric_list = {'HR', 'RESP', 'SpO2', 'ABPMean', 'NBPMean'};
duration = 3600;
feature_path = sprintf('../data/f_428.0_%d.csv', duration);

if extract_feature_flag
  extract_feature_of(id_list, feature_path, metric_list,duration);
end

n_metrics = length(metric_list);
n_feature_per_metric = 3;
n_feature_for_class = 1;
comb = combnk(1:n_feature_per_metric, n_feature_for_class);

loss_array = zeros(1, n_metrics * n_feature_per_metric);
condition_cell_array = cell(1, n_metrics * n_feature_per_metric);

for m_idx = 1:n_metrics
  for c_idx = 1:size(comb,1)
    f_idx = comb(c_idx,:) + (m_idx-1) * n_feature_per_metric;
    display(m_idx);
    display(f_idx);
    
    loss = classify_feature(feature_path, mortality_flag, f_idx);
    display(loss);
    
    loss_array((m_idx-1) * n_feature_per_metric + c_idx) = loss;
    condition_cell_array{(m_idx-1) * n_feature_per_metric + c_idx} = f_idx;
  end
end

[min_loss, index] = min(loss_array);
display(sprintf('Min_cond: %s Value: %f', mat2str(condition_cell_array{index}), min_loss));
display_feature(feature_path, mortality_flag, condition_cell_array{index})

