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