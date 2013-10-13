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

%grad_l_exp = min(realmax, exp(-bsxfun(@times, Y, bsxfun(@times,w,X))));
%grad_l_nom = bsxfun(@times, bsxfun(@times,Y,X),grad_l_exp);
%grad_l = sum(grad_l_nom ./ (ones(size(grad_l_exp)) + grad_l_exp),1);
grad_l = [];
for i = 1 : size(X,1)
    grad_exp = min(realmax, exp(-Y(i) * w * X(i,:)'));
    m = Y(i) * X(i,:) .* grad_exp /(1 + grad_exp);
    grad_l = [grad_l ; m];
end
grad = sum(grad_l,1) - C * w;
