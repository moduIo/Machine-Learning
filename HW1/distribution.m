total = 0;              % Tracks total of current iteration
limit = 10000;          % Total number of iterations
means = zeros(limit);   % Array of means for plotting
count = (1:limit);      % Array of [1..limit] for plotting

for i = 1:limit
    for j = 1:i
        total = total + randn;
    end
    
    mean = total / i;
    means(i) = mean;
    total = 0;
end

figure;
plot(count, means);