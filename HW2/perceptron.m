%
% Implements the PLA running until convergence.
% The data is assumed to be linearly seperable.
%
function [w,w_0] = train_perceptron(X, y)
    [m, d] = size(X);     % It is assumed that X does not have x_0 = 1 already
    one = ones(m, 1);     % Make x_0 = 1 vector
    X_bar = [one X];      % Append vector to input X
    w = zeros(d + 1, 1);  % Initialize w = 0 vector
    
    while ~converged(w, X_bar, y, m)
        for i = 1:m
            if dot(w, X_bar(i, :)) * y(i) <= 0
                w = w + (y(i) * X_bar(i, :)');    
            end
        end
    end
    
    % Remove bias term from w
    w_0 = w(1);
    w = w(2:length(w));
end

%
% Function iterates over all training examples to determine
% if PLA has converged.
%
function done = converged(w, X, y, m)
    for i = 1:m
        if dot(w, X(i, :)) * y(i) <= 0
            done = 0;
            return;
        end
    end
    
    done = 1;
    return;
end