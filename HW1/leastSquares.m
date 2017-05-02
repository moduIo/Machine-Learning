function [w, w_0] = train_ls(X, y, bias)
    dim = size(X);
    
    % Add a row of all 1's for bias term
    if bias
        inX = zeros(dim(1), dim(2) + 1);
        for i = 1:dim(1)
            inX(i, :) = [1 X(i, :)];
        end
    else
        inX = X;
    end
    
    % Check for invertibility
    if det(inX.' * inX) == 0
        [V, D] = eig(inX.' * inX);
        D_plus = zeros(size(D));
        
        % Create the positive D matrix as discussed in lecture slides
        for i = 1:(size(D))(1)
            for j = 1:(size(D))(2)
                if D(i, j) ~= 0
                    D_plus(i, j) = 1/D(i, j);
                end
            end
        end
        
        w = V * D_plus * V.' * inX.' * y;
        
    % Otherwise return the normal solution
    else
        w = inv(inX.' * inX) * inX.' * y;
    end
    
    % If a bias term is used in X then we need to move the value
    % from w to w_0
    if bias
        w_0 = w(1);
        w = w(2:length(w));
    else
        w_0 = 0;
    end 
end