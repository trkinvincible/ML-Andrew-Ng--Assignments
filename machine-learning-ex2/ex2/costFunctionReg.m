function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n=length(theta);
z = X * theta;
Hx = sigmoid(z);
J = (-(1/m)*sum(((y'*log(Hx)) + ((1-y)'*log(1-Hx))))) + lambda/(2*m)*sum(theta(2:n).^2);

%Do not regularize theta(1)
grad(1)= (1/m) * X'(1,:) * (Hx-y); %Do not sum because "*" on 2 Vectors already do SUM

%(1/m) * (X' * (Hx-y))

% below is partial derivative so cannot use costfunction directly
for j=2:n
	grad(j)= (1/m) * (X'(j,:) * (Hx-y)) + (lambda/m)*theta(j);
end
% =============================================================

end
