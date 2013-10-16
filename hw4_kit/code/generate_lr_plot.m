%% Load the dataset.
load ../data/breast-cancer-data.mat

%% Part I - Learning Rates
clear answers

% Your plot of objective vs. iteration for learning rates should use the
% following step sizes:
step_size_range = [1 0.1 0.001 0.0001 1e-5];

% YOUR CODE GOES HERE
iter_range = 1 :1: 5000;
w_set = [];
grad_set = [];
obj_set = cell(1, size(step_size_range));
train_err = zeros(1,size(step_size_range));

for j = 1:length(step_size_range)
   [w,obj,gradnorm] = lr_train(X,Y,0.001,'step_size',step_size_range(j),'max_iter',5000,'stop_tol',0);    
   [t] = lr_test(w,X);
   train_err(j) = length(find(t~=Y))/length(Y);
   obj_set{j} = obj;
   w_set = [w_set w];
   grad_set = [grad_set gradnorm];
end
hold on;
set(gca,'YScale','log');
ylim([-10^6,-10^1]);
plot(iter_range,obj_set{1},'y', iter_range,obj_set{2},'r',iter_range,obj_set{3},'b',iter_range,obj_set{4},'g',iter_range,obj_set{5},'m');
legend('step=1','step=0.1','step=0.001','step=0.0001','step=1e-5');
hold off;
% Save with:
print -djpeg -r72 step_sizes.jpg

answers{1} = '';

answers{2} = '';

save('answers_1.mat', 'answers');

% %% Part II - Learning Curves
clear answers

% For this section, generate the learning curves. Make sure to plot
% errorbars.

% YOUR CODE GOES HERE
nb_test_err = zeros(1,8);
nb_test = zeros(100,8);

lr_test_err = zeros(1,8);
lr_test = zeros(100,8);
%(a) Randomly separate the dataset into 80% training, 20% test

for i = 1 : 100
    part = (mod(randperm(length(Y)), 5) + 1)';
    train_x = X(part < 5,:);
    train_y = Y(part < 5,:);
    test_x = X(part == 5,:);
    test_y = Y(part == 5,:);

    %(b) Further subdivide the training set into 8 partitions
    partition =  (mod(randperm(length(train_x)), 8) + 1)';
   
    for j = 1 : 8      
        
        nb_test_err(:,j) = nb_learning(partition, j, X, Y, test_x, test_y);       
        lr_test_err(:,j) = lr_learning(partition, j, X, Y, test_x, test_y);
    end
    
    nb_test(i,:) = nb_test_err;
    lr_test(i,:) = lr_test_err;
end
nb_test_err_final = mean(nb_test);
nb_test_std_final = std(nb_test);
lr_test_err_final = mean(lr_test);
lr_test_std_final = std(lr_test);
% plot
axis_x = 1 : 8;
errorbar(axis_x, nb_test_err_final, nb_test_std_final, 'b');
xlabel('Datapoints');
ylabel('Test Error');
title('Learning curves');
hold on;
errorbar(axis_x, lr_test_err_final, lr_test_std_final, 'r');

legend('blue: Naive Bayes', 'red: Logistic Regression');
print -djpeg -r72 learning_curves.jpg;
hold off;
% 
% answers{1} = ''; 
% 
% save('answers_2.mat', 'answers');
