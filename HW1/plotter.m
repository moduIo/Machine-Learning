train = [4.701929e+09, 4.422658e+09, 4.073035e+09, 3.971484e+09, 3.951972e+09];
test = [6.040003e+09, 6.079690e+09, 6.288992e+09, 6.323187e+09, 8.611349e+09];
t = 1:5;
plot(t, test),
xlabel('k'), ylabel('Square Loss')
title('Test Loss')