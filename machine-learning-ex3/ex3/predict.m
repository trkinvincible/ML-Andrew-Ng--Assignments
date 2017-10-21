function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
l2_units=size(Theta1);
%disp(l2_units);
l3_units=size(Theta2);
%disp(l3_units);

% Add ones to the X data matrix
X = [ones(m, 1) X];

%(z layer 1)
z1 = (X * Theta1');
%disp(size(za1));%5000x25
%(activation layer 1)
a1 = sigmoid(z1);

%a1 = [ones(m,1) a1];
disp(size(a1));

z2 = (a1 * Theta2');
%disp(size(za2));%5000x10
a2 = sigmoid(z2);

for i=1:m
   [x, ix] = max(a2(i,:));
   p(i)=ix;
end









% =========================================================================


end
