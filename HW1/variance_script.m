total = 0;                 % Tracks total of current iteration
limit = 10000;             % Total number of iterations
variances = zeros(limit);  % Array of means for plotting
count = (1:limit);         % Array of [1..limit] for plotting

for i = 1:limit
    samples = (1:i);  % Holds the samples for the current experiement
    
    % Generate each sample
    for j = 1:i
       samples(j) = randn;
    end
    
    % Calculate empirical mean of sample
    for j = 1:i
        total = total + samples(j);
    end
    
    mean = total / i;
    total = 0;
    
    sum = 0;  % Total of the summation of variance
    
    % Calculate summation of variance
    for j = 1:i
        sum = sum + (samples(j) - mean)^2;
    end
    
    variance = (1 / (i - 1)) * sum;
    variances(i) = variance;
end

figure;
plot(count, variances);