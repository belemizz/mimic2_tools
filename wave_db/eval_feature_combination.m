csvdata = csvread('../data/428.0.csv');
id_list = csvdata(1,:);
mortality_flag = csvdata(2,:);
feature_path = '../data/test_feature.csv';

n_metrics = 3;
feature_per_metric = 3;
comb = combnk(1:feature_per_metric, 2);

loss_array = zeros(1, n_metrics * feature_per_metric);

for m_idx = 1:n_metrics
  for c_idx = 1:length(comb)
    f_idx = comb(c_idx,:) + (m_idx-1) * feature_per_metric;
    loss = classify_feature(feature_path, mortality_flag, f_idx);
    loss_array((m_idx-1) * feature_per_metric + c_idx) = loss;
  end
end
