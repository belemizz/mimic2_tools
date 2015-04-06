% Comparison before/after excluding unreliable data points

id_list = [1931, 12581, 12920, 2157];
metric_list = {'HR', 'RESP', 'SpO2','NBPMean', 'ABPMean'};
save_graph = true;
duration = 10800;

draw_graph_of(id_list, metric_list, save_graph, duration, false);
draw_graph_of(id_list, metric_list, save_graph, duration, true);
