%
% Trains algorithms and plots hyperplanes
%
function empirical_observations(Xtest, Xtrain, ytest, ytrain, perceptron)
    % Split training data into classes
    [d, m] = size(ytrain);
    pos = [];
    neg = [];
    
    for i = 1: d
        if ytrain(i) == 1
            pos = [pos; Xtrain(i,:)];
        else
            neg = [neg; Xtrain(i,:)];
        end
    end
    
    % Train algorithms
    if perceptron 
        [w_perceptron, w_0_perceptron] = train_perceptron(Xtrain, ytrain);
    end
    [w_rr, w_0_rr] = train_rr(Xtrain, ytrain, 1);
    [w_svm_01, w_0_svm_01] = train_svm_primal(Xtrain, ytrain, 0.01);
    [w_svm_100, w_0_svm_100] = train_svm_primal(Xtrain, ytrain, 10000);

    % Test algorithms
    
    if perceptron 
        loss_perceptron = test_algorithm(Xtest, ytest, w_perceptron, w_0_perceptron); 
    end
    loss_rr = test_algorithm(Xtest, ytest, w_rr, w_0_rr);
    loss_svm_01 = test_algorithm(Xtest, ytest, w_svm_01, w_0_svm_01);
    loss_svm_100 = test_algorithm(Xtest, ytest, w_svm_100, w_0_svm_100);

    % Display empirical errors
    if perceptron
        disp('Perceptron Loss ');
        disp(loss_perceptron);
    end
    
    disp('Ridge Regression Loss ');
    disp(loss_rr);
    
    disp('SVM C = .01 Loss ');
    disp(loss_svm_01);
    disp('SVM C = .01 Support Vector Count ');
    disp(count_support_vectors(Xtrain, w_svm_01, w_0_svm_01));
    
    disp('SVM C = 100 Loss ');
    disp(loss_svm_100);
    disp('SVM C = 100 Support Vector Count ');
    disp(count_support_vectors(Xtrain, w_svm_100, w_0_svm_100));
    
    % Define hyperplanes
    if perceptron 
        a_p = -1 * w_perceptron(1)/w_perceptron(2);
        b_p = -1 * w_0_perceptron/w_perceptron(2);
        x_p = -5:10;
        y_p = a_p * x_p + b_p;
    end
    
    a_rr = -1 * w_rr(1)/w_rr(2);
    b_rr = -1 * w_0_rr/w_rr(2);
    x_rr = -5:10;
    y_rr = a_rr * x_rr + b_rr;

    a_svm_01 = -1 * w_svm_01(1)/w_svm_01(2);
    b_svm_01 = -1 * w_0_svm_01/w_svm_01(2);
    x_svm_01 = -5:10;
    y_svm_01 = a_svm_01 * x_svm_01 + b_svm_01;

    a_svm_100 = -1 * w_svm_100(1)/w_svm_100(2);
    b_svm_100 = -1 * w_0_svm_100/w_svm_100(2);
    x_svm_100 = -5:10;
    y_svm_100 = a_svm_100 * x_svm_100 + b_svm_100;
    
    % Plot points
    figure,
    scatter(pos(:,1), pos(:,2), 25, 'b'),
    hold on,
    scatter(neg(:,1), neg(:,2), 25, 'r'),
    %plot(x_p, y_p),
    plot(x_svm_01, y_svm_01),
    plot(x_svm_100, y_svm_100),
    plot(x_rr, y_rr),
    hold off,
    legend('Positive', 'Negative', 'PLA', 'SVM C = .01', 'SVM C = 100', 'RR', 'Location', 'southeast');
end

%
% Calculates the empirical risk of trained hypothesis
%
function loss = test_algorithm(X, y, w, w_0)
    [m, d] = size(X);  % Size of input matrix
    loss = 0;          % Value of loss
    
    % Sum over each training example
    for i = 1:m
        if y(i) * (dot(w, X(i, :)) + w_0) <= 0
            loss = loss + 1;
        end
    end
    
    loss = loss / m;
end

%
% Calculates the number of support vectors
%
function count = count_support_vectors(X, w, b)
    [m, d] = size(X);  % Size of input matrix
    count = 0;         % Number of support vectors
    
    % Calculate the margin of each sample
    for i = 1:m
        margin = abs(dot(w, X(i, :)) + b);
        if margin <= 1.0001 && margin >= .9999
            count = count + 1;
        end
    end
end

%------------------------------------------------
function [w,w_0] = train_rr(X, y, lambda)
    [m, d] = size(X);  % It is assumed that X does not have x_0 = 1 already
    I = eye(d + 1);    % Make the identity matrix
    one = ones(m, 1);  % Make x_0 = 1 vector
    X_bar = [one X];  % Append vector to input X
    
    % Compute minimum weights
    w = inv((X_bar' * X_bar) + (lambda * I)) * (X_bar' * y);
    
    % Remove bias term from w
    w_0 = w(1);
    w = w(2:length(w));
end

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

function [w,w_0] = train_svm_primal(X, y, C)
    [m, d] = size(X);
    
    % The first d + 1 represents w and w_0 while the last m are the slack variables
    H = diag([ones(1, d), zeros(1, m + 1)]);
    
    % The f term only affects the slack variables
    f = [zeros(1, d + 1), C * ones(1, m)];
    
    % Break up the constraints into parts
    A = -1 * [diag(y) * X, y, eye(m)];
    b = -1 * ones(m, 1);
    
    % There are no linear constraints
    Aeq = [];
    beq = [];
    
    % Constrain slack variables to be greater than or equal to 0
    lb = [-1 * inf(1, d + 1), zeros(1, m)];
    ub = inf(1, d + 1 + m);
    
    solution = quadprog(H, f, A, b, Aeq, beq, lb, ub);
    w = solution(1:d);
    w_0 = solution(d + 1);
end