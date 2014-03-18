% 2.2
function test()
clear all; clc;

Q = 2;
T = 6;

N = [5 5 1 1 1 0]';
M = [5 5 1 1 1 0 20;
    3 3 3 1 4 13 20];
pq = [2/3 1/3]';
cg = [7 9 10 13 19 30];
%cg = 1*ones(T,1);
ca = 18;

% vars x_{ij}, W_{qi}, S_{qi}
x = zeros((T),(T+1)); lx0 = 0; lxf = T*(T+1);
W = zeros(Q,T); lW0 = lxf; lWf = lW0 + Q*T;
S = zeros(Q,(T+1)); lS0 = lWf; lSf = lS0 + Q*(T+1);

vars = (T)*(T+1)+Q*T+Q*(T+1);
cons = (T)+Q+Q*(T+1)+Q;

    function id = x_id(i,j)
        id = (i-1)*(T+1)+j;
    end
    function id = W_id(q,i)
        id = lW0+ (q-1)*T+i;
    end
    function id = S_id(q,i)
        id = lS0+ (q-1)*(T+1)+i;
    end

% cons 1
A1 = zeros(T,vars);
b1 = N;
for i=1:T
    for j=i+1:T+1
        A1(i,x_id(i,j)) = 1;
    end
end

% cons 2
b2 = sum(M,2);
A2 = zeros(Q,vars);
for q=1:Q
    for i=1:T+1
        A2(q, S_id(q,i)) = 1;
    end
end

% cons 3
b3 = zeros(Q*(T+1),1);
for q=1:Q
    for i=1:(T+1)
        which_row = (q-1)*(T+1)+i;
        b3(which_row) = -M(q,i);
    end
end
A3 = zeros(Q*(T+1),vars);
for q=1:Q
    for i=1:(T+1)
        which_row = (q-1)*(T+1)+i;
        for j=1:i
            A3(which_row, x_id(j,i)) = -1;
        end
        if i == 1
            A3(which_row, W_id(q,i)) = 1;
        elseif i == T+1
            A3(which_row, W_id(q,i-1)) = -1;
        else
            A3(which_row, W_id(q,i)) = 1;
            A3(which_row, W_id(q,i-1)) = -1;
        end
        A3(which_row, S_id(q,i)) = -1;
    end
end

% cons 4
lb = zeros(vars,1);
ub = 1000*ones(vars,1);

Aeq = [A1;A2;A3];
beq = [b1;b2;b3];

In = 1:vars;
e = 10e-6;

% value function
f = zeros(vars,1);
for i=1:T
    for j=i+1:(T+1)
        f(x_id(i,j)) = cg(j-i);
    end
    for q=1:Q
        f(W_id(q,i)) = ca*pq(q);
    end
end


%[xans,fval] = IP(f, [],[], Aeq, beq, lb,[], In, e);
[xans,fval] = linprog(f, [],[], Aeq, beq, lb,ub);


fval
x = reshape(xans(1:lxf),T,(T+1))
W = reshape(xans(lW0+1:lWf),Q,T)
%S = reshape(xans(lS0+1:lSf),Q,T+1)

a =5
end