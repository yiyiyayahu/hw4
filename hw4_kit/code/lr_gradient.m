function [grad] = lr_gradient(X, Y, w, C)
% Compute the Logistic Regression gradient.
%
% Usage:
%
%    [GRAD] = LR_GRADIENT(X, Y, W, C)
%
% X is a N x P matrix of N examples with P features each. Y is a N x 1 vector
% of (-1, +1) class labels. W is a 1 x P weight vector. C is the regularization
% parameter. Computes the gradient w.r.t. W of the regularized logistic
% regression objective and returns a 1 x P vector GRAD.
%
% SEE ALSO
%   LR_TRAIN, LR_TEST

% YOUR CODE GOES HERE
grad_l = [];
for i = 1 : size(X,1)
    grad_exp = exp(-Y(i) * w * X(i,:)');
    if grad_exp == Inf
        grad_exp = realmax;
    end
    m = Y(i) * X(i,:) .* grad_exp ./(1 + grad_exp);
    grad_l = [grad_l ; m];
end
grad = sum(grad_l,1) - C * w;
grad(grad == Inf) = realmax;
