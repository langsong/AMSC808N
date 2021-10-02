%% I collaborated with Zhirui Li on the inplementation of the GD and SG algorithms
%% Main driver

%the global minimum
as = (3* cos(pi/5)+cos(3*pi/10) - cos(2*pi/5))/ pi; 
bs = (1.4 * pi *a_prob1-4+cos(pi/5)+cos (3* pi /10)+cos (2* pi / 5 ) ) / 4;
% disp(f(as,bs));

pa = fp(1,0);
pb = fb(1,0);
% disp(pa);
% disp(pb);
alpha_s = (pi/2)/(-pb + (pi/2) *pa);

%Call gradient descend function
[a_list,b_list] = graddes(0.99*alpha_s);

%Show the result
close all
figure(1)
hold on
xlabel("iter");
ylabel("value");
plot(1:100001, a_list, 1:100001, b_list)
legend('a', 'b')


%Run the SGD algorithm
[iter, a_list,b_list] = sgrad(0.5);

%Show the result
figure(2)
hold on
xlabel("iter");
ylabel("value");
plot(1:iter+1, a_list, 1:iter+1, b_list)
legend('a', 'b')

%% definition of functions and algorithms
function y = g(x)
    y = 1 - cos(x);
end

function z = fp(a,b) %gradient of f over a
    x1 = 0;
    x2 = pi/10;
    x3 = 2*pi/10;
    x4 = 3*pi/10;
    x5 = 4*pi/10;
    x6 = 5*pi/10;
    z = (1/6)*((x1*a - b>0)*x1*(x1*a - b - g(x1)) + (x2*a - b>0)*x2*(x2*a - b - g(x2)) + (x3*a - b>0)*x3*(x3*a - b - g(x3)) + (x4*a - b>0)*x4*(x4*a - b - g(x4)) + (x5*a - b>0)*x5*(x5*a - b - g(x5)) + (x6*a - b>0)*x6*(x6*a - b - g(x6)));
end

function z = fb(a,b) %gradient of f over b
    x1 = 0;
    x2 = pi/10;
    x3 = 2*pi/10;
    x4 = 3*pi/10;
    x5 = 4*pi/10;
    x6 = 5*pi/10;
    z = (1/6)*((x1*a - b>0)*(-1)*(x1*a - b - g(x1)) + (x2*a - b>0)*(-1)*(x2*a - b - g(x2)) + (x3*a - b>0)*(-1)*(x3*a - b - g(x3)) + (x4*a - b>0)*(-1)*(x4*a - b - g(x4)) + (x5*a - b>0)*(-1)*(x5*a - b - g(x5)) + (x6*a - b>0)*(-1)*(x6*a - b - g(x6)));
end

function[gradlossa,gradlossb] = gradtrain(j,a,b)  %function used for SGD algorithm
x1 = 0;
x2 = pi/10;
x3 = 2*pi/10;
x4 = 3*pi/10;
x5 = 4*pi/10;
x6 = 5*pi/10;
if j == 0
    gradlossa = (x1*a - b - g(x1)).*x1.*(x1*a - b>0);
    gradlossb = (x1*a - b>0)*(-1)*(x1*a - b - g(x1));
elseif j == 1
    gradlossa = (x2*a - b - g(x2)).*x2.*(x2*a - b>0);
    gradlossb = (x2*a - b>0)*(-1)*(x2*a - b - g(x2));
elseif j == 2
    gradlossa = (x3*a - b - g(x3)).*x3.*(x3*a - b>0);
    gradlossb = (x3*a - b>0)*(-1)*(x3*a - b - g(x3));
elseif j == 3
    gradlossa = (x4*a - b - g(x4)).*x4.*(x4*a - b>0);
    gradlossb = (x4*a - b>0)*(-1)*(x4*a - b - g(x4));
elseif j == 4
    gradlossa = (x5*a - b - g(x5)).*x5.*(x5*a - b>0);
    gradlossb = (x5*a - b>0)*(-1)*(x5*a - b - g(x5));
else
    gradlossa = (x6*a - b - g(x6)).*x6.*(x6*a - b>0);
    gradlossb = (x6*a - b>0)*(-1)*(x6*a - b - g(x6));
end
end

function t = f(a,b) %the loss function f(a,b)
    x1 = 0;
    x2 = pi/10;
    x3 = 2*pi/10;
    x4 = 3*pi/10;
    x5 = 4*pi/10;
    x6 = 5*pi/10;
    t = (1/12)*((max(0, (x1*a - b)) - g(x1))^2 + (max(0, (x2*a - b)) - g(x2))^2 + (max(0, (x3*a - b)) - g(x3))^2 + (max(0, (x4*a - b)) - g(x4))^2 + (max(0, (x5*a - b)) - g(x5))^2 + (max(0, (x6*a - b)) - g(x6))^2);
end

%gradient descent algorithm
function [a_list, b_list] = graddes(alpha)
a = 1;
b = 0;
max_iter = 100000;
iter = 0;
a_list = zeros(1,max_iter);
b_list = zeros(1,max_iter);
while iter < max_iter && norm([fp(a,b),fb(a,b)]) > 1e-8
    iter = iter + 1;
    a_list(iter) = a;
    b_list(iter) = b;
    grad_a = fp(a,b);
    grad_b = fb(a,b);
    a = a - alpha*grad_a;
    b = b - alpha*grad_b;
end
a_list(iter+1) = a;
b_list(iter+1) = b;
a_list = a_list(1:(iter+1));
b_list = b_list(1:(iter+1));
end

%stochastic gradient descent algorithm
function [iter, a_list,b_list] = sgrad(alpha0) 
    a=1;
    b=0;
    max_iter = 100000;
    r_size = 150; 
    iter = 1; 
    a_list = zeros(1,max_iter);
    b_list = zeros(1,max_iter);
    a_list(iter) = a;
    b_list(iter) = b; 
    iter = iter +1;
    j = randi([0,5]);
    [grad_a, grad_b] = gradtrain(j,a,b);
    a = a - alpha0*grad_a;
    b = b - alpha0*grad_b;
    a_list(iter) = a;
    b_list(iter) = b;
    while iter < max_iter && (norm(a_list(iter)-a_list(iter-1), b_list(iter)-b_list(iter-1)) > 1e-11)
        iter = iter + 1;
        if iter <= r_size  %for the first r_size iterations, use initial stepsize
            alpha = alpha0;
        elseif iter <= 2*r_size  %for the next r_size iterations, use alpha0/2 stepsize
            alpha = alpha0/2;
        else  %if current iteration is greater than 2*r_size
            cur_iter = iter - 2*r_size;
            index = 2:40;
            factor = idivide(int64(2.^index), int64(index));
            round_length = r_size.*cumsum(factor);  %thresholds that determine when to reduce stepsize
            round_index = find(round_length > cur_iter, 1, 'first');  %find the corresponding round index the current iteration is on
            alpha = alpha0/(2^(round_index+1));
        end
        j = randi([0,5]);
        [grad_a, grad_b] = gradtrain(j,a,b);
        a = a - alpha*grad_a;
        b = b - alpha*grad_b;
        a_list(iter) = a;
        b_list(iter) = b;
    end
    a_list(iter+1) = a;
    b_list(iter+1) = b;
    a_list = a_list(1:(iter+1));
    b_list = b_list(1:(iter+1));
    
    disp(['Result: ', 'a= ', num2str(a), ', b= ', num2str(b)]);
    disp(['Loss= ',  num2str(f(a,b))]);
end