function [w,w_0] = train_rr(X, y, lambda)
    [m, d] = size(X);  % It is assumed that X does not have x_0 = 1 already
    I = eye(d + 1);    % Make the identity matrix
    one = ones(m, 1);  % Make x_0 = 1 vector
    X_bar = [one X];  % Append vector to input X
    
    display(X_bar);
    
    % Compute minimum weights
    w = inv((X_bar' * X_bar) + (lambda * I)) * (X_bar' * y);
    
    % Remove bias term from w
    w_0 = w(1);
    w = w(2:length(w));
end