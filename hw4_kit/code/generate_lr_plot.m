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
% clear answers
% 
% % For this section, generate the learning curves. Make sure to plot
% % errorbars.
% 
% % YOUR CODE GOES HERE

% 
% 
% % Save with:
% print -djpeg -r72 learning_curves.jpg
% 
% answers{1} = ''; 
% 
% save('answers_2.mat', 'answers');
