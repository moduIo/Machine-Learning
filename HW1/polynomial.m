% Driver function for polynomial linear regression
function polynomial(X_train, X_test, Y_train, Y_test, k)
    % Generate polynomial features
    poly_train = generate_poly_features(X_train, k);
    poly_test = generate_poly_features(X_test, k);
    
    % Normalize data
    [norm_poly_train, norm_poly_test] = normalizeAll(poly_train, poly_test);
    
    % Train using least squares
    [w, w_0] = train_ls(norm_poly_train, Y_train, true);
    
    % Calculate the in sample errors
    trainLoss = squareError(w, w_0, norm_poly_train, Y_train);
    testLoss = squareError(w, w_0, norm_poly_test, Y_test);
    
    % Output errors
    fprintf('Training Error: %d\n', trainLoss);
    fprintf('Test Error: %d\n', testLoss);
end

% Function nomalizes training and test sets
function [X_train_norm, X_test_norm] = normalizeAll(X_train, X_test)
    trainMax = max(X_train);              % Maximum of training data features
    trainMin = min(X_train);              % Minimum of training data features
    testMax = max(X_test);                % Maximum of test data features
    testMin = min(X_test);                % Minimum of test data features
    trainSize = size(X_train);            % Dimensions of training data
    testSize = size(X_test);              % Dimensions of test data
    X_train_norm = zeros(size(X_train));  % Normalized training set
    X_test_norm = zeros(size(X_test));    % Normalized test set
    
    % Normalize the training data
    for i = 1:trainSize(1)
        for j = 1:trainSize(2)
            X_train_norm(i, j) = normalize(X_train(i, j), trainMax(j), trainMin(j));
        end
    end
    
    % Normalize the test data
    for i = 1:testSize(1)
        for j = 1:testSize(2)
            X_test_norm(i, j) = normalize(X_test(i, j), testMax(j), testMin(j));
        end
    end
end

% Function normalizes a datapoint w.r.t. the max and min elements
function y = normalize(x, maximum, minimum)
    y = 2 * ((x - minimum)/(maximum - minimum)) - 1;
end

% Function generates poylnomial features to the degree k from input matrix
function [X_poly] = generate_poly_features(X, k)
    length = size(X);                        % Original dimensions of X
    pDimensions = length(2) * k;             % Number of polynomail dimensions to be generated
    X_poly = zeros(length(1), pDimensions);  % Polynomial feature matrix
    
    % For every sample
    for m = 1:length(1)
        % For each dimension
        for d = 1:length(2)
            power = 1;  % Current power
            
            % Expand the feature k times
            for i = ((d - 1) * k) + 1:((((d - 1) * k) + 1) + (k - 1))
                X_poly(m, i) = X(m, d)^power;
                power = power + 1;
            end
        end
    end
end

% Calculates sample error of X on y w.r.t. w and w0
function loss = squareError(w, w0, X, y)
    dimensions = size(X);  % Size of input matrix
    m = dimensions(1);     % Number of samples
    loss = 0;              % Value of loss
    
    % Sum over each training example
    for i = 1:m
        loss = loss + (y(i) - (w0 + dot(w, X(i,:))))^2;
    end
    
    loss = loss / m;
end

%------------------------------
% Numerically solves least squares
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