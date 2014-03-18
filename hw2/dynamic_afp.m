function dynamic_afp()
clear all; clc;

Q = 5;
T = 6;
S = 6;
F = 2;
dep = [1 3]';
arr = [3 4]';
cg = 200; ca=1000;

% branch stuff
B = 8;
omg = zeros(B,5);
omg(1,1) = 1;
omg(2,2:end) = ones(4,1)';
omg(3,2) = 1;
omg(4,3:end) = ones(3,1)';
omg(5,3) = 1;
omg(6,4:end) = ones(2,1)';
omg(7,4) = 1;
omg(8,5) = 1;

o = [1 1 2 2 3 3 4 4]';
mu = [5 1 5 2 5 3 5 5]'; 
    
N = zeros(S,T);
N(1,3) = 1;
N(3,4) = 1;
M = [0 1 1 2 2 2 2;
    0 0 1 1 2 2 2;
    0 0 0 1 1 2 2;
    0 0 0 0 1 1 2;
    0 0 0 0 0 1 2];
p = [0.01 0.48 0.48 0.02 0.01]';

X = zeros(Q,F,T+1); lX0 = 0; lXf = Q*F*(T+1);
Y = zeros(Q,F,T+1); lY0 = lXf; lYf = lY0 + Q*F*(T+1);
W = zeros(Q,T); lW0 = lYf; lWf = lW0 + Q*T;

vars = 2*Q*F*(T+1)+ Q*T;

    function id = X_id(q,f,t)
        id = (t-1)*Q*F + (q-1)*F+f;
    end
    function id = Y_id(q,f,t)
        id = lY0 + (t-1)*Q*F + (q-1)*F+f;
    end
    function id = W_id(q,t)
        id = lW0+ (q-1)*T+t;
    end

% cost function
func = zeros(vars,1);
for q=1:Q
    for f=1:F
        for t=arr(f):T+1
            func(X_id(q,f,t)) = cg*p(q)*(t-arr(f));
        end
    end
    for t=1:T
        func(W_id(q,t)) = p(q)*ca;
    end
end

% cons1, eq3
b1 = ones(F*Q,1);
A1 = zeros(F*Q,vars);
for f=1:F
    for q=1:Q
        which_row = (f-1)*Q + q;
        for t=arr(f):T+1
            A1(which_row, X_id(q,f,t)) = 1;
        end
    end
end

% cons2, eq4
b2 = zeros((T+1)*Q,1);
A2 = zeros((T+1)*Q, vars);
for t=1:T+1
    for q=1:Q
        which_row = (t-1)*Q+q;
        b2(which_row) = M(q,t);
        
        for f=1:F
            A2(which_row, X_id(q,f,t)) = 1;
        end
        if t == 1
            A2(which_row, W_id(q,t)) = -1;
        elseif t==T+1
            A2(which_row, W_id(q,t-1)) = 1;
        else
            A2(which_row, W_id(q,t)) = -1;
            A2(which_row, W_id(q,t-1)) = 1;
        end
    end
end


%cons3, eq5
% neglect this?

%cons4, eq7
A4 = [];
b4 = [];
for f=1:F
    for t=1:T
        for i=1:B
            if (o(i) <= t) && (t <= mu(i))
                Ni = sum(omg(i,:));
                if Ni >= 2
                    for k1=1:Q
                        if omg(i,k1) == 1
                            for k2=1:Q
                                if omg(i,k2) == 1
                                    row = zeros(vars,1)';
                                    row(Y_id(k1,f,t)) = 1;
                                    row(Y_id(k2,f,t)) = 1;
                                    
                                    b4 = [b4;0];
                                    A4= [A4;row];
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

%cons5, eq1
b5 = [];
A5 = [];
for f=1:F
    for t=dep(f):T
        for q=1:Q
            row = zeros(vars,1)';
            row(Y_id(q,f,t)) = 1;
            row(X_id(q,f,t+arr(f)-dep(f))) = -1;
            
            A5 = [A5;row];
            b5 = [b5; 0];
        end
    end
end

lb = zeros(vars,1);
ub = 100*ones(vars,1);
ub(1:lYf) = 1;

In = 1:vars;
e = 10e-6;

Aeq = [A1;A4;A5];
beq = [b1;b4;b5];

Aieq = [A2];
bieq = [b2];

[xans,fval] = IP(func, Aieq, bieq, Aeq, beq, lb, ub, In, e);
fval
Xans = zeros(Q,F,T+1);
for q=1:Q
    for f=1:F
        for t=1:T+1
            Xans(q,f,t) = xans(X_id(q,f,t));
            if xans(X_id(q,f,t)) > 0
                sprintf('(%d,%d,%d)=%d',q,f,t,xans(X_id(q,f,t)))
            end
        end
    end
end
W = reshape(xans(lW0+1:lWf),T,Q)'
z = 5
end