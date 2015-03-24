%% Test 1: draw graph
id_list = sort(csvread('../data/out.csv'));
id_select = id_list(2:3);
metric_list = {'HR','SpO2'};
save_graph = true;
draw_graph_of(id_select, metric_list, save_graph);

%% Test 2: extract feature
id_list = sort(csvread('../data/out.csv'));
id_select = id_list(2:3);
extract_feature_of(id_select, 'first');

