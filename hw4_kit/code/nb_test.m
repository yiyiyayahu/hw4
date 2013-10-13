function [y] = nb_test(nb, X)
% Generate predictions for a Gaussian Naive Bayes model.
%
% Usage:
%
%   [Y] = NB_TEST(NB, X)
%
% X is a N x P matrix of N examples with P features each, and NB is a struct
% from the training routine NB_TRAIN. Generates predictions for each of the
% N examples and returns a 0-1 N x 1 vector Y.
% 
% SEE ALSO
%   NB_TRAIN

% YOUR CODE GOES HERE (compute log_p_x_and_y)
x_and_y_1 = zeros(size(X));
x_and_y_2 = zeros(size(X));
for i = 1 : size(x_and_y_1,2);
    x_and_y_1(:,i) = log(normpdf(X(:,i),nb.mu_x_given_y(i,1),nb.sigma_x(i)).^2);
    x_and_y_2(:,i) = log(normpdf(X(:,i),nb.mu_x_given_y(i,2),nb.sigma_x(i)).^2);
end
log_p_x_and_y = [bsxfun(@plus,log(1-nb.p_y),sum(x_and_y_1,2)) bsxfun(@plus,log(nb.p_y),sum(x_and_y_2,2))];
% Take the maximum of the log generative probability 
[~, y] = max(log_p_x_and_y, [], 2);
% Convert from 1,2 based indexing to the 0,1 labels
y = y -1;



    
