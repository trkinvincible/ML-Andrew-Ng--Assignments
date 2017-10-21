function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Add ones to the X data matrix
X = [ones(m, 1) X];

%(z layer 1)
z2 = (X * Theta1');
%disp(size(z2));%5000x25
%(activation layer 2)
a2 = sigmoid(z2);

a2 = [ones(size(a2,1),1) a2];
%disp(size(a2));

z3 = (a2 * Theta2');
%disp(size(z3));%5000x10
a3 = sigmoid(z3);

Y = zeros(m,num_labels);
%Y is a matrix of 5000*k with every row is with only the bit turned ON for corresponding y
for i=1:m
  Y(i,y(i)) = 1;%refer Page 5 of ex4.pdf
end
%disp(size(Y));

%why .*  here? because we need only the correctly predicted classifier's data for cost calculation
#{
Y is like below with only correct y bit turned on.
1 1...k
..
..
m 1...k

#}
J = -1 * (1/m)*sum(sum((Y .* log(a3)) + ((1-Y) .* log(1-a3))));
%disp(J);

#{
  sum will by default add along the column resulting in single row
  if you want to sum along row do sum(a,1);
  below do not include theta of baised unit
#}
R = lambda/(2*m)* (sum(sum((Theta1(:,2:size(Theta1,2)).^2))) + sum(sum(Theta2(:,2:size(Theta2,2)).^2)));

J = J + R;
%disp(J);

for t = 1:m
  #{
        there are 4 steps to be done as per ex4.pdf
  #}
%STEP - 1(forward propogation)

	%take each pixel data (1 row at a time) input layer
	a1 = [X(t,:)'];

	% For the hidden layers, where l=2:
	z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  %add biased unit
	a2 = [1;a2];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

%STEP - 2(find how much each unit at that layer responsible for the error)

  %create a vector with only y(t) valued elements '1' others '0'
	yy = ([1:num_labels]==y(t))';
	%error at L3 is straight forward just compare Hx with actual output from training set
	EPSILON_3 = a3 - yy;
  
%STEP - 3(find how much each unit in hidden layer responsible for the error)

  %for hidden layer apply formula as stated in the ex4.pdf
	EPSILON_2 = (Theta2' * EPSILON_3) .* [1; sigmoidGradient(z2)];
  
  %remove error for biased unit
	EPSILON_2 = EPSILON_2(2:end);

	% EPSILON_1 is not calculated because we do not associate error with the input    
%STEP -4 (Accumulate the gradient from this example)

	% Big EPSILON update as per formula in ex4.pdf
	Theta1_grad = Theta1_grad + EPSILON_2 * a1';
	Theta2_grad = Theta2_grad + EPSILON_3 * a2';
  
end
%STEP - 5(now divide the obtained delta with no. of training examples)

%you should not be regularizing the  first column of theta

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
