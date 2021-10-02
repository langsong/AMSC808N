as = (3* cos(pi/5)+cos(3*pi/10) - cos(2*pi/5))/ pi;
bs = (1.4 * pi *a_prob1-4+cos(pi/5)+cos (3* pi /10)+cos (2* pi / 5 ) ) / 4;
% disp(f(as,bs));


pa = fp(1,0);
pb = fb(1,0);
% disp(pa);
% disp(pb);
alpha_s = (pi/2)/(-pb + (pi/2) *pa);

%Call gradient descend function
[as,bs] = graddes(0.99*alpha_s);
close all
figure(2)
hold on
xlabel("iter");
ylabel("value");
plot(1:100001, as, 1:100001, bs)
legend('a', 'b')

for j = 1:10
    msgmain = ['Begin ', num2str(j), 'th SGD simulation'];
    disp(msgmain)
    sgrad()
    disp('SGD finished')
    disp('-------')
end

%%plots
% close all
% figure(1)
% hold on
% yl = [-4,4];
% xl = [-4,4];
% xBox = [0, xl(1), xl(1), 0, 0];
% yBox = [0 ,0, yl(2), yl(2), 0];
% patch(xBox,yBox,'red' , 'FaceColor' , 'green');
% xs = 0:0.5:10;
% line1 = pi * xs / 2 ;
% xTri = [0,0 ,10/pi , 0] ;
% yTri = [0 ,yl(2) , yl(2) , 0];
% patch( xTri , yTri , 'red' , 'FaceColor' , 'green' ) ;
% a_prob1 = (3* cos(pi/5)+cos(3*pi/10) - cos(2*pi/5))/ pi;
% b_prob1 = (1.4 * pi *a_prob1-4+cos(pi/5)+cos (3* pi /10)+cos (2* pi / 5 ) ) / 4;
% scatter ( a_prob1 , b_prob1 , "filled")
% xlabel (" a ")
% ylabel ("b")

c = -5:1:5;
d = c;
[a,b] = meshgrid(c);
x1 = 0;
x2 = pi/10;
x3 = 2*pi/10;
x4 = 3*pi/10;
x5 = 4*pi/10;
x6 = 5*pi/10;

%%definition of functions
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

function[gradlossa,gradlossb] = gradlosstrain(j,a,b)
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

function t = f(a,b) %the loss function
    x1 = 0;
    x2 = pi/10;
    x3 = 2*pi/10;
    x4 = 3*pi/10;
    x5 = 4*pi/10;
    x6 = 5*pi/10;
    t = (1/12)*((max(0, (x1*a - b)) - g(x1))^2 + (max(0, (x2*a - b)) - g(x2))^2 + (max(0, (x3*a - b)) - g(x3))^2 + (max(0, (x4*a - b)) - g(x4))^2 + (max(0, (x5*a - b)) - g(x5))^2 + (max(0, (x6*a - b)) - g(x6))^2);
end

function [alist, blist] = graddes(alpha) %gradient descent algorithm
%     for alpha = alphalist
%         dmsg = ['Current alpha = ', num2str(alpha)];
%         disp(dmsg);
        a = 1;
        b = 0;
        max_iter = 100000;
        iter = 0;
        alist = nan(max_iter, 1);
        blist = nan(max_iter, 1);
        while iter < max_iter && norm([fp(a,b),fb(a,b)]) > 1e-8
            iter = iter + 1;
            alist(iter) = a;
            blist(iter) = b;
            grad_a = fp(a,b);
            grad_b = fb(a,b);
            a = a - alpha*grad_a;
            b = b - alpha*grad_b;
        end
        disp([a,b]);
        disp(f(a,b));
        disp([fp(a,b), fb(a,b)]);
        disp(iter);
        disp('----------------------------');
        
        alist(iter+1) = a;
        blist(iter+1) = b;
        alist = alist(1:(iter+1));
        blist = blist(1:(iter+1));
    %end
end

function [] = sgrad() %stochastic gradient descent algorithm
    a=1;
    b=0;
    alpha0 = 0.5;
    max_iter = 100000;
    iter = 1; 
    alist = zeros(1,max_iter);
    blist = zeros(1,max_iter);
    alist(iter) = a;
    blist(iter) = b;
    m0 = 150; 
    iter = iter +1;
    alpha = alpha0;
    j = randi([0,5]);
    [grad_a, grad_b] = gradlosstrain(j,a,b);
    a = a - alpha*grad_a;
    b = b - alpha*grad_b;
    alist(iter) = a;
    blist(iter) = b;
    while iter < max_iter && (norm(alist(iter)-alist(iter-1), blist(iter)-blist(iter-1)) > 1e-11)
        iter = iter + 1;
        if iter <=m0
            alpha = alpha0;
        elseif iter <= 2*m0
        alpha = alpha0/2;
        else
            index = iter - 2*m0;
            klist = 2:40;
            seq = idivide(int64(2.^klist), int64(klist));
            indexint = m0.*cumsum(seq);
            kk = find(indexint > index, 1, 'first');
            alpha = alpha0/(2^(kk+1));
        end
        j = randi([0,5]);
        [grad_a, grad_b] = gradlosstrain(j,a,b);
        a = a - alpha*grad_a;
        b = b - alpha*grad_b;
        alist(iter) = a;
        blist(iter) = b;
    end
    alist(end) = a;
    blist(end) = b;
    disp(['Result: ', 'a= ', num2str(a), ', b= ', num2str(b)]);
    disp(['Loss= ',  num2str(f(a,b))]);
end