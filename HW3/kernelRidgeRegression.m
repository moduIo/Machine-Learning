%
% Implementation of Kernel Ridge Regression
%
function kernelRidgeRegression(Xtr, ytr, Xte, yte, lambda, kernel, gamma)
    alpha = train_krr(Xtr, ytr, lambda, kernel, gamma);
    ypredicted = test_krr(alpha, Xtr, ytr, Xte, lambda, kernel, gamma);
    
    % Calculate prediction accuracy
    [m, ~] = size(yte);
    misclassified = 0;
    
    for i = 1: m
        if yte(i) ~= ypredicted(i)
            misclassified = misclassified + 1;
        end
    end
    
    accuracy = double(length(yte) - misclassified)/length(yte);
    
    disp('Test Accuracy: ');
    disp(accuracy);
end

%
% Calculates alpha using the solution of 5b
%
function alpha = train_krr(X, y, lambda, kernel, gamma)
    [m, ~] = size(X);  % Dimensions of data
    kX = zeros(m, m);  % Holds the value of the kernel
    
    if strcmp('gaussian', kernel)
        % Construct each element of XX^T via kernel function application
        for i = 1: m
            for j = 1: m
                kX(i, j) = exp((-1 * gamma) * norm(X(i,:) - X(j, :))^2);
            end
        end
        
        alpha = inv((1/(m * lambda)) * kX + eye(m)) * y;
        
    elseif strcmp('linear', kernel)
        % Directly compute alpha
        alpha = inv((1/(m * lambda)) * (X * X') + eye(m)) * y;
    end   
end

%
% Computes predicted values of test samples
%
function ypredicted = test_krr(alpha, Xtr, ytr, Xte, lambda, kernel, gamma)
    [m, d] = size(Xtr);    % Dimensions of training data
    [mte, ~] = size(Xte);  % Dimensions of test data
    w = d:1;               % Weight vector
    ypredicted = mte:1;    % Prediction vector
    
    if strcmp('gaussian', kernel)
        % Compute predictions
        for i = 1: mte
            
            ker = zeros(1, m);
            
            for j = 1: m
                ker(j) = exp((-1 * gamma) * norm(Xtr(j,:) - Xte(i, :))^2);
            end
            
            if (1/(m * lambda)) * ker * alpha > 0
                ypredicted(i) = 1;
            else
                ypredicted(i) = -1;
            end
        end
        
        % Compute training error
        misclassified = 0;

        for i = 1: m
            ker = zeros(1, m);
            
            for j = 1: m
                ker(j) = exp((-1 * gamma) * norm(Xtr(j,:) - Xtr(i, :))^2);
            end
            
            if ytr(i) * (1/(m * lambda)) * ker * alpha < 0
                misclassified = misclassified + 1;
            end
        end
        
    elseif strcmp('linear', kernel)
        w = (1/(m * lambda)) * Xtr' * alpha;
        
        % Compute predictions
        for i = 1: mte
            if Xte(i, :) * w > 0
                ypredicted(i) = 1;
            else
                ypredicted(i) = -1;
            end
        end
        
        % Compute training error
        misclassified = 0;

        for i = 1: m
            if ytr(i) * Xtr(i, :) * w < 0
                misclassified = misclassified + 1;
            end
        end
    end

    % Output accuracy
    accuracy = double(m - misclassified)/m;
    disp('Training Accuracy: ');
    disp(accuracy); 
end