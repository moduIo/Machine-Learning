function w = incremental_train_ls(X, y)
    [m, d] = size(X);     % Dimensions of data
    inc_X = zeros(m, d);  % Used to ensure that inc_X is invertible
    w = zeros(1, d);         % Weights to be learned
    
    % Ensures that inc_X is invertible
    inc_X(1:d, 1:d) = 10^-4 * eye(d);
    
    % Calculate the first inverse
    inc_X(1, :) = X(1, :);
    sm_inv = inv(inc_X.' * inc_X);  % Hold the inverse
    w = sm_inv * inc_X.' * y;
    
    if m > 1
        % Incrementally update the weights
        for i = 2:m
            next = X(i, :);  % The next training example
            inc_X(i, :) = next;  % Add next into X
            next = next.';  % Convert into column vector
            
            % Compute SM inverse
            top = sm_inv * (next * next.') * sm_inv;
            bottom = 1 + (next.' * sm_inv * next);
            sm_inv = sm_inv - (top / bottom);
            
            % Compute OLS optimization
            w = sm_inv * inc_X.' * y;
        end
    end
end