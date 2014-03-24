import cvxpy as cp
from numpy import *

#def static_afp():
"""
T=6
Q = 2
N = [5, 5, 1, 1, 1, 0]
M = [[5, 5, 1, 1, 1, 0, 20],
    [3, 3, 3, 1, 4, 13, 20]]
p = [0.666, 0.333]
cg = [7,9,10,13,19,30]
ca = 18
"""

Q=5
T=6
N = [0,0,1,1,0,0]
M = [[0,1,1,2,2,2,2],
    [0,0,1,1,2,2,2],
    [0,0,0,1,1,2,2],
    [0,0,0,0,1,1,2],
    [0,0,0,0,0,1,2]]
p = [0.01, 0.48, 0.48, 0.02, 0.01]
cg = [200 for i in xrange(T)]
ca = 1000


x = cp.Variable(T,T+1)
W = cp.Variable(Q,T)
S = cp.Variable(Q,T+1)

#vars = T*(T+1) + Q*T + Q*(T+1)
#cons = T + Q + Q*(T+1)

f = 0*x[0,0]
for i in xrange(T):
    for j in xrange(i+1,T+1):
        f = f + cg[j-i-1]*x[i,j]
for q in xrange(Q):
    for i in xrange(T):
        f = f + ca*p[q]*W[q,i]

# cons1
c1 = []
for i in xrange(T):
    c1.append(sum([x[i,j] for j in xrange(i+1,T+1)]) == N[i])

# cons 2
c2 = []
for q in xrange(Q):
    c2.append(sum([S[q,i] for i in xrange(T+1)]) ==
        sum(M[q][i] for i in xrange(T+1)))

# cons 3
c3 = []
tmp_W_qi = cp.Variable()
tmp_W_qim1 = cp.Variable()
for q in xrange(Q):
    for i in xrange(T+1):
        if i == T:
            tmp_W_qi = 0
            tmp_W_qim1 = W[q,i-1]
        elif i == 0:
            tmp_W_qi = W[q,i]
            tmp_W_qim1 = 0
        c3.append(sum([-x[h,i] for h in xrange(i)]) -tmp_W_qim1
                +tmp_W_qi - S[q,i] == -M[q][i])

# cons 3
c4 = [0<= x, x<=1, 0<=W, W<= 1, 0<=S]

cons = c1+c2+c3+c4

p = cp.Problem(cp.Minimize(f), cons)
res = p.solve()
print res
print x.value