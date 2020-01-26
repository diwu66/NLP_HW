%% "quadprog" https://www.mathworks.com/help/optim/ug/quadprog.html
% input
Y = [-1;-1;1;1];
X = [0 0; 2 2; 2 0; 3 0];

% process
[n, p0] = size(X);  %n:sample size;
p = p0+1;           %p:feature size (include intercept b)

Q = [0 0 0; 0 1 0; 0 0 1];  
P = zeros(p,1);
A = - diag(Y) * horzcat(ones(n,1), X);  
c = - ones(n,1);

% run quadprog
[x,fval,exitflag,output,lambda] = quadprog(Q,P,A,c);
% Confirm that Q is positive definite by checking its eigenvalues
eig(Q)
%{ 
Examine the final point, function value, and exit flag 
An exit flag of 1 means the result is a local minimum. Because Q is a positive definite matrix, this problem is convex, so the minimum is a global minimum.)
x,fval,exitflag
x =
   -1.0000
    1.0000   
   -1.0000
fval =
    1.0000
exitflag =   
    1 
%}

% predict samples--> proportions of correct samples (correct if satisfies A*x <= c)
sum(A*x <= c)/n   % 1 = all correct

