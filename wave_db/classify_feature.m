function loss = classify_feature(feature_file, flag)
% plot feature with flag
% flag 1: red
% flag 0: blue

display_feature(feature_file, flag);
figure;
%svmStruct = svmtrain(feature_file, flag, 'ShowPlot', true);

feature_seq = csvread(feature_file);

SVMModel = fitcsvm(feature_seq, flag);
CVSVMModel = crossval(SVMModel);
loss = kfoldLoss(CVSVMModel);

end
