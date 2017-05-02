%
% Function plots iterations of k-means via metric
%
function plot(X, y, k_max, T, metric)
    k = 2;
    val = [];  % Holds each iterations value
    
    while k <= k_max
        M = kmeans(X, k, T);  % Compute centers
        
        if strcmp(metric, 'Sum of Squares')
            val = [val sum_of_squares(M, X, k)]; 
        elseif strcmp(metric, 'Purity')
            val = [val, purity(M, X, y, k)];
        else
            disp('Invalid metric');
            return
        end
        
        k = k + 2;  % Iterate by 2
    end
    
    % Plot values
    plot(val);
end

%
% Function computes purity
%
function purity = purity(M, X, y, k)
    [m, ~] = size(X);      % Dimensions of training data
    C = zeros(m, 1);       % Entries are {1, ..., k} for sample cluster
    labels = zeros(1, k);  % Holds cluster labels by majority vote
    
    % Assign each sample to nearest center
    for i = 1: m
        C(i) = compute_nearest_center(M, X(i, :), k);
    end
    
    % Assign labels
    for i = 1: k
        dist = zeros(1, 10);  % Distribution of labels in cluster i
        
        for j = 1: m
            
            % If sample is in cluster add the label to dist
            if C(j) == i
                dist(y(j) + 1) = dist(y(j) + 1) + 1;
            end
        end
        
        % Store index of max class
        [~, l] = max(dist);
        labels(i) = l - 1;  % Class labels are from 0-9
    end
    
    error = 0;
    
    % Compute error
    for i = 1: m
        % If the label is different than the cluster label
        if labels(C(i)) ~= y(i)
           error = error + 1; 
        end
    end
    
    % Defined as the proportion of correct labels
    purity = (m - error) / m;
end

%
% Function computes sum of squares
%
function sum = sum_of_squares(M, X, k)
    [m, ~] = size(X);
    C = zeros(m, 1);  % Entries are {1, ..., k} for sample cluster
    sum = 0;
    
    % Assign each sample to nearest center
    for i = 1: m
        C(i) = compute_nearest_center(M, X(i, :), k);
    end
    
    % Calculate squared distance for each cluster
    for i = 1: m
        sum = sum + norm(X(i, :) - M(C(i), :))^2;
    end
end

%
% Function computes the set of k centers via k-means clustering
%
function M = kmeans(X, k, T)
    [m, d] = size(X);   % Dimensions of data
    M = zeros(k, d);    % Matrix of k centers
    C = zeros(m, 1);    % Entries are {1, ..., k} for sample cluster
    
    % Sequentially assign centers as first k examples
    for i = 1: k
        M(i, :) = X(i, :);
    end
    
    % Iterate a max of T times
    for i = 1: T
        converged = true;  % Tracks if algorithm has converged
        
        % Assign each sample to nearest center
        for j = 1: m
            prev = C(j);  % Track previous center
            C(j) = compute_nearest_center(M, X(j, :), k);
            
            if C(j) ~= prev
                converged = false;
            end
        end
        
        % Terminate on convergence
        if converged
            break;
        end
        
        % Recompute centers
        compute_centers(M, C, X);
    end
end

%
% Function returns the index of the closest center to sample
%
function c = compute_nearest_center(M, x, k)
    c = 0;             % Index of nearest center
    closest = intmax;  % Current value closest distance
    
    for i = 1: k
        dist = norm(x - M(i, :));  % Compute distance via l2 norm
        
        if dist < closest
            c = i;
            closest = dist;
        end
    end
end

%
% Function recomputes centers as average vector of cluster
%
function compute_centers(M, C, X)
    [k, d] = size(M);
    [m, ~] = size(X);
    
    % Recompute center for each cluster
    for i = 1: k
        center = zeros(1, d);  % Average vector
        m_cluster = 0;         % Number of samples in cluster
        
        for j = 1: m
            if C(j) == i
                center = center + X(i, :);
                m_cluster = m_cluster + 1;
            end
        end
        
        M(i, :) = center ./ m_cluster;
    end
end