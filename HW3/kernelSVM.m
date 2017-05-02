%
% Main.
% All input is expected to be mxd dimensional
% Ex: driver(Xtr', ytr', Xte', yte', 10, 'linear', .001)
%
function driver(Xtr, ytr, Xte, yte, C, kernel, gamma)
    alpha = train_ksvm_dual(Xtr, ytr, C, kernel, gamma);
    ypredicted = test_ksvm_dual(alpha, Xtr, ytr, Xte, kernel, gamma);
    
    % Calculate prediction accuracy
    [m, d] = size(yte);
    misclassified = 0;
    
    for i = 1: m
        if yte(i) ~= ypredicted(i)
            misclassified = misclassified + 1;
        end
    end
    
    accuracy = double(length(yte) - misclassified)/length(yte);
    
    disp('Accuracy: ');
    disp(accuracy);
    
    % Count support vectors
    supports = 0;
    [alpha_m, alpha_d] = size(alpha);
    
    for i = 1: alpha_m
        if alpha(i) > C/100
            supports = supports + 1;
        end
    end
    
    disp('Support Vectors: ');
    disp(supports);
end

%
% Trains on kernel SVM dual objective.
%
function [alpha, value] = train_ksvm_dual(X, y, C, kernel, gamma)
    [m, d] = size(X);
    
    % K is the mxm matrix of the kernel function applied as k(x_i, x_j)
    if strcmp('gaussian', kernel)
        K = zeros(m, m);
        
        % Compute K(x_i, x_j) for each i, j
        for i = 1:m
            for j = 1:m
                K(i, j) = exp((-1 * gamma) * norm(X(i, :) - X(j, :))^2);
            end
        end
    elseif strcmp('linear', kernel)
        K = X * X';
    end
    
    % So H is composed of y_i * y_j * k(x_i, x_j)
    H = (y * y') .* K;
    f = -1 * ones(m, 1);
    
    % There are no constraints
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    
    % Using bounds on alpha to handle inequality constraints
    lb = zeros(m, 1);
    ub = C * ones(m, 1);
    
    [alpha, value] = quadprog(H, f, A, b, Aeq, beq, lb, ub);
    
    disp('Objective value: ');
    disp(value);
end

%
% Generates predicted y via learned w for each x.
%
function ypredicted = test_ksvm_dual(alpha, Xtr, ytr, Xte, kernel, gamma)
    [m, d] = size(Xtr);  % Dimensions of train
    w = zeros(1, d);   % Weight vector
    [mte, dte] = size(Xte);  % Dimensions of test
    ypredicted = 1:mte;  % Output vector
    
    % Compute w
    if strcmp('gaussian', kernel)
        % For each test sample
        for i = 1: mte
            % Compute the predicted value of the test
            prediction = 0;
            
            for j = 1: m
                prediction = prediction + (alpha(j) * ytr(j) * exp((-1 * gamma) * norm(Xtr(j, :) - Xte(i, :))^2));
            end
            
            if prediction > 0
                ypredicted(i) = 1;
            else
                ypredicted(i) = -1;
            end
        end
        
    elseif strcmp('linear', kernel)
        % Use formulation from Q1
        for i = 1: m
            w = w + (alpha(i) * ytr(i) * Xtr(i, :));
        end
            
        for i = 1: mte
            if Xte(i, :) * w' > 0
                ypredicted(i) = 1;
            else
                ypredicted(i) = -1;
            end
        end
    end
end