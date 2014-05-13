from gurobipy import *
import numpy as np
from cutstock import *
import pdb

s,B = create_example(random=False, eg=2)

n = len(s)
U = len(get_bound(s,B))
model = Model('cutstock')
setParam('TimeLimit', 10)

# 1. add vars
#   x[i,j] = 1 if i^th demand falls into j^th bin
#   y[j] = 1 if j^th bin is used 
x,y = {}, {}
for i in xrange(n):
    for j in xrange(U):
        x[i,j] = model.addVar(vtype='B', name='x[%d,%d]'%(i,j))

for j in xrange(U):
    y[j] = model.addVar(obj=1, vtype='B', name='y[%d]'%j)
model.update()

# 2. constraints
for i in xrange(n):
    vars = [x[i,j] for j in xrange(U)]
    coeffs = [1]*U
    model.addConstr(LinExpr(coeffs, vars), '=', 1, name='c1[%d]'%i)

for j in xrange(U):
    vars = [x[i,j] for i in xrange(n)]
    coeffs = [s[i] for i in xrange(n)]
    model.addConstr(LinExpr(coeffs, vars), '<', LinExpr(B, y[j]), name='c2[%d]'%j)

for i in xrange(n):
    for j in xrange(U):
        model.addConstr(x[i,j], '<', y[j], name='c3[%d,%d]'%(i,j))

model.update()
pdb.set_trace()
