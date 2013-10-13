function [y] = lr_test(w, X)
% Generate predictions for a logistic regression model.
%
% Usage:
%
%   [Y] = LR_TEST(W, X)
%
% X is a N x P matrix of N examples with P features each. W is a 1 x (P+1) 
% vector of weights returned by LR_TRAIN. The output, Y, is a N x 1 vector
% of 0-1 class labels predictions.
%
% SEE ALSO
%   LR_TRAIN, LR_GRADIENT

% Add a constant feature to each example to learn a bias term.
X = [ones(size(X,1), 1) X];

% Compute P(Y|X):
% YOUR CODE GOES HERE
p_y = findPre(X,w);
% Convert P(Y|X) to predictions.
y = p_y>=0.5;
end

function [p_y] = findPre(X,w)
    o = exp(- X * w');
    o_exp_1 = min(o, realmax * ones(size(o)));
    p_y = ones(size(o_exp_1)) ./ (ones(size(o_exp_1)) + o_exp_1);
end
