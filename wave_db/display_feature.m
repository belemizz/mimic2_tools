function display_feature(feature_file, flag, index)
% plot feature with flag
% flag 1: red
% flag 0: blue

feature_seq = csvread(feature_file);
feature_of_int = feature_seq(:,index);

%display(feature_of_int);

red_feature = feature_of_int(flag==1, :);
blue_feature = feature_of_int(flag==0, :);

% % regular graph
figure;
plot(red_feature(:,1), red_feature(:,2), 'r.', 'MarkerSize', 20);
hold on;
plot(blue_feature(:,1), blue_feature(:,2), 'b.', 'MarkerSize', 20);
xlabel('feature 1');
ylabel('feature 2');

% semilog graph
figure;
semilogy(red_feature(:,1), red_feature(:,2), 'r.', 'MarkerSize', 20);
hold on;
semilogy(blue_feature(:,1), blue_feature(:,2), 'b.', 'MarkerSize', 20);
xlabel('feature 1');
ylabel('feature 2');

% loglog graph
figure;
loglog(red_feature(:,1), red_feature(:,2), 'r.', 'MarkerSize', 20);
hold on;
loglog(blue_feature(:,1), blue_feature(:,2), 'b.', 'MarkerSize', 20);
xlabel('feature 1');
ylabel('feature 2');

end
