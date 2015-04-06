icd9_code = '428.0_427.31';
signal_duration = 10800;

id_path = sprintf('../data/%s.csv', icd9_code);
csvdata = csvread(id_path);

id_list = csvdata(1,:);
mortality_flag = csvdata(2,:);

index = 1:5;

metric_list = {'HR', 'RESP', 'SpO2'};

%% preprocesses
mortality_id = id_list(mortality_flag == 1);
non_mortality_id = id_list(mortality_flag == 0);

for idx = 1:length(metric_list)
  draw_graph_of(mortality_id(index), display_metric(idx), true, signal_duration);
  draw_graph_of(non_mortality_id(index), display_metric(idx), true, signal_duration);
end
