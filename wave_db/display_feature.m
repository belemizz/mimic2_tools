function display_feature(feature_file, flag)
% plot feature with flag
% flag 1: red
% flag 0: blue

feature_seq = csvread(feature_file);
display(feature_seq);

red_feature = feature_seq(flag==1, :);
blue_feature = feature_seq(flag==0, :);

% regular graph
figure;
plot(red_feature(:,1), red_feature(:,2), 'r.', 'MarkerSize', 20);
hold on;
plot(blue_feature(:,1), blue_feature(:,2), 'b.', 'MarkerSize', 20);
xlabel('feature 1');
ylabel('feature 2');

% semilog graph
% figure;
% semilogy(red_feature(:,1), red_feature(:,2), 'r.', 'MarkerSize', 20);
% hold on;
% semilogy(blue_feature(:,1), blue_feature(:,2), 'b.', 'MarkerSize', 20);
% xlabel('feature 1');
% ylabel('feature 2');

end
