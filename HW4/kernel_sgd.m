%
% TODO: Implement driver function
%

%
% Function returns the accuracy of the prediction
%
function [correct, misclassified] = test_prediction(ypred, yte)
    [m, ~] = size(yte);
    correct = zeros(10, 1);         % Correct predictions per class
    misclassified = zeros(10, 10);  % (correct class, prediction)
    
    for i = 1: m
        % If the prediction was wrong store prediction
        if ypred(i) ~= yte(i)
            misclassified(yte(i) + 1, ypred(i) + 1) = misclassified(yte(i) + 1, ypred(i) + 1) + 1;
        else
            correct(ypred(i) + 1) = correct(ypred(i) + 1) + 1;
        end
    end
    
    disp('Accuracy:');
    disp(sum(correct)/m);
end

%
% Function computes the predicted classes
%
function [ypred] = test_mhinge_kernel_sgd(alpha, Xsv, Xte, p)
    [m, d] = size(Xte);
    [t, k] = size(alpha);
    ypred = zeros(m, 1);
    
    % Predict for each test example
    for i = 1: m
        max = intmin;  % Determine most likely class
        pred = 0;
        
        % Precompute the kernel function
        ker = kernel(Xte(i, :), Xsv, p);
        
        % Calculate scores for each class predictor
        for class = 1: k
            score = 0;
            
            % Compute <w, x>
            for j = 1: t
                score = score + (alpha(j, class) * ker(j));
            end

            % Update prediction
            if score > max
                max = score;
                pred = class;
            end
        end
        
        ypred(i) = pred - 1;  % Classes range from 0-9, k from 1-10
    end
end

%
% Function computes kernel SGD
%
function [alpha, Xsv] = train_mhinge_krnel_sgd(Xtr, ytr, Delta, p)
    [m, d] = size(Xtr);     % Dimensions of data
    k = 10;                 % Number of classes
    alpha = zeros(1, k);    % Matrix of alphas
    Xsv = zeros(1, d);      % Saved samples
    
    % Bootstrap the first example
    max = 0;
    yhat = 0;
    
    % Find yhat
    for i = 1: k
        loss = Delta(i, ytr(1));
        
        if loss > max
            max = loss;
            yhat = i;
        end
    end
    
    Xsv(1, :) = Xtr(1, :);  % Initialize Xsv to only the first example
    
    % Compute alpha_{j, i} where eta_1 = 1
    for i = 1: k
        if i == ytr(1) + 1
            alpha(1, i) = 1;
        elseif i == yhat
            alpha(1, i) = -1;
        else
            alpha(1, i) = 0;
        end
    end
    
    % Compute SGD over remaining training examples
    for t = 2: m
        y = ytr(t) + 1;  % Label for current training sample
        x = Xtr(t, :);   % Current sample
        cur_alpha = zeros(1, k);  % Current alpha
        eta = 1/sqrt(t);  % Learning rate
        
        % Determine yhat
        yhat = maximize_loss(alpha, Xsv, Delta, k, x, y, p);
        
        % Update Alpha
        for i = 1: k
            if i ~= yhat && i == y
                cur_alpha(1, i) = eta;
            
            elseif i == yhat && i ~= y
                cur_alpha(1, i) = -1 * eta;
            
            % Otherwise either y = yhat = i or y ~= i and yhat ~= i
            else 
                cur_alpha(1, i) = 0;
            end
        end
        
        % If an update occured
        if any(cur_alpha)
            alpha = [alpha; cur_alpha];
            Xsv = [Xsv; x];
        end
    end
end

%
% Function computes loss at timestep
%
function loss = compute_loss(alpha, y, ker)
    loss = 0;   % Initial loss
    [t, ~] = size(alpha);
    
    for i = 1: t
        loss = loss + (alpha(i, y) * ker(i));
    end
end

%
% Function returns class which maximizes loss
%
function yhat = maximize_loss(alpha, Xsv, Delta, k, x, y, p)
    max = intmin;  % Maximum loss value
    yhat = 0;      % Maximum loss label
    
    % Precompute the kernel function
    ker = kernel(x, Xsv, p);
    
    % Compute loss term involving y_t
    y_loss = compute_loss(alpha, y, ker);
    
    % Find y' which maximizes loss
    for i = 1: k
        if i == y
            loss = Delta(i, y);
        else
            % Compute loss term involving y'
            yhat_loss = compute_loss(alpha, i, ker);
            loss = Delta(i, y) + yhat_loss - y_loss;
        end

        % Update maximizer
        if loss > max
            max = loss;
            yhat = i;
        end
    end
end

%
% Function computes polynomial kernel
%
function ker = kernel(x, Xsv, p)
    [m, ~] = size(Xsv);
    ker = zeros(m, 1);
    
    for i = 1: m
        ker(i) = (x * Xsv(i, :)')^p;
    end
end

%
% Returns the number of different entries between 2 matrices
%
function diff = difference(A, B)
    [m, ~] = size(A);
    diff = 0;
    
    for i = 1: m
        if ~ all(A(i, :) == B(i, :))
            diff = diff + 1;
        end
    end
end