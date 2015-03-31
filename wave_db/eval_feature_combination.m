csvdata = csvread('../data/428.0.csv');
id_list = csvdata(1,:);
mortality_flag = csvdata(2,:);
feature_path = '../data/test_feature.csv';

extract_feature_flag = false;
metric_list = {'HR', 'RESP', 'SpO2', 'ABPMean', 'NBPMean'};

if extract_feature_flag
  extract_feature_of(id_list, feature_path, metric_list);
end

n_metrics = 1;
feature_per_metric = 4;
comb = combnk(1:feature_per_metric, 2);

loss_array = zeros(1, n_metrics * feature_per_metric);

for m_idx = 1:n_metrics
  for c_idx = 1:length(comb)
    f_idx = comb(c_idx,:) + (m_idx-1) * feature_per_metric;
    display(m_idx);
    display(f_idx);
    
    loss = classify_feature(feature_path, mortality_flag, f_idx);
    display(loss);
    
    loss_array((m_idx-1) * feature_per_metric + c_idx) = loss;
  end
end
