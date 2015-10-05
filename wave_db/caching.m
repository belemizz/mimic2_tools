id_list = sort(csvread('../data/out.csv'));
id_select = id_list(1);
metric_list = {'HR','SpO2'};

save_data_of(id_select, metric_list)