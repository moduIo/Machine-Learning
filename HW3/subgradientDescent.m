%
% Calculates various observations from data
%
function plot(Xte, yte, Xtr, ytr, C, T, a)
    [w] = train_svm_sgd(Xtr, ytr, C, a, T);
    [~, T] = size(w);
    train = calculateError(w, Xtr, ytr);
    test = calculateError(w, Xte, yte);
    objective = calculateObjective(w, Xtr, ytr, C);
    
    % Plot objective function
    figure,
    loglog(objective');
    
    % Plot training error
    figure,
    plot(1:T, train);

    % Plot test error
    figure,
    plot(1:T, test);
end

%
% Calculates objective value
%
function value = calculateObjective(w, X, y, C)
    [~, T] = size(w);     % Dimensions of weights
    [m, d] = size(X);     % Dimensions of X
    value = zeros(T, 1);  % Value for each iteration
    
    % Calculate objective value for each w_t
    for t = 1: T
        sum = 0;
        
        % Calculate summation term
        for i = 1: m
            sum = sum + max(0, 1 - (y(i) * (X(i, :) * w(:, T))));
        end
        
        value(t) = (.5 * norm(w(:, t))^2) + (C * sum);
    end    
end

%
% Calculates empirical error
%
function error = calculateError(w, X, y)
    [~, T] = size(w);     % Dimensions of weights
    [m, ~] = size(X);     % Dimensions of X
    error = zeros(T, 1);  % Error for each iteration
    
    % Calculate empirical error for each w_t
    for t = 1: T
        loss = 0;
        
        for i = 1: m
            if y(i) * (X(i, :) * w(:, t)) < 0
                loss = loss + 1;
            end
        end
        
        error(t) = loss/m;
    end
end

%
% Trains linear SVM using subgradient descent
% All input is expected to be mxd dimensional
% Ex: train_svm_sgd(Xtr', ytr', 10, 1, 100)
%
function [w] = train_svm_sgd(X, y, C, a, T)
    [m, d] = size(X);   % Dimensions of data
    w_t = zeros(d, 1);  % Initialize w_0 = 0
    w = w_t;            % w is a dx(T + 1) matrix of w_t from each timestep
    
    % Run algorithm T iterations
    for t = 1: T
        eta = a/t;  % Weight decay
        sum = 0;    % Misclassified sum
        
        % Calculate summation of misclassified
        for i = 1: m
            if y(i) * (X(i, :) * w_t) <= 1
                sum = sum + y(i) * X(i, :)';
            end
        end
        
        w_t = ((1 - eta) * w_t) + (eta * C * sum);
        
        w = [w w_t];  % Return each w_t in w
    end
end