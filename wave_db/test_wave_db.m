%% Test 1:Draw full length graph
id_list = sort(csvread('../data/out.csv'));
id_select = id_list(1);
metric_list = {'HR','SpO2'};
save_graph = true;
draw_graph_of(id_select, metric_list, save_graph);

%% Test 2:Draw graph with fixed length
id_list = sort(csvread('../data/out.csv'));
id_select = id_list(2:3);
metric_list = {'HR','SpO2','RESP','NBPMean','ABPMean'};
save_graph = true;
length_of_data = 3600;
draw_graph_of(id_select, metric_list, save_graph, length_of_data);

%% Test 3:Extract feature
id_list = sort(csvread('../data/out.csv'));
id_select = id_list(2:3);
extract_feature_of(id_select, 'first');

%% Test 4:Feature mapping
csvdata = csvread('../data/428.0.csv');
id_list = csvdata(1,1:10);
mortality_flag = csvdata(2,1:10);
feature_path = '../data/test_feature.csv';
extract_feature_of(id_list, 'first',feature_path);
display_feature(feature_path, mortality_flag);

%% Test 5:SVM Classification
csvdata = csvread('../data/428.0.csv');
id_list = csvdata(1,1:10);
mortality_flag = csvdata(2,1:10);
feature_path = '../data/test_feature.csv';
loss = classify_feature(feature_path, mortality_flag);
display(loss);

