function loss = classify_feature(feature_file, flag, index)
% plot feature with flag
% flag 1: red
% flag 0: blue

display_feature(feature_file, flag);
%svmStruct = svmtrain(feature_file, flag, 'ShowPlot', true);

feature_seq = csvread(feature_file);
feature_of_int = feature_seq(:,index);

SVMModel = fitcsvm(feature_of_int, flag);
CVSVMModel = crossval(SVMModel);
loss = kfoldLoss(CVSVMModel);

end
