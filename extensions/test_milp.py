from cvxpy import *
from mixed_integer import *

x = BoolVar()
y = BoolVar()

f = 0.6*x+ 0.2*y
c1 = x >= 0
c2 = y >= 0
c3 = 2*x + 2*y <= 6
c4 = x+y >= 1

p = Problem(Minimize(f), [c1,c2,c3,c4])
res = p.solve(method="admm")
#res = p.solve()

print res
