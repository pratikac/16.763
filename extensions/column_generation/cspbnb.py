from gurobipy import *
import numpy as np
import pdb
import timer
import branch_and_bound

eps = 1e-6
LOG = False

def create_instance(n=40):
    '''
        B: roll width
        w: orders
        q: quantity of w[i]
    '''
    #np.random.seed(3)
    B = 100
    s = [0]*n 
    for i in xrange(n):
        s[i] = np.random.randint(1,99)
    return s,B


def get_bound(s,B):
    '''
    use greedy algorithm to get bound on maximum # of bins
    '''
    remain = [B]
    soln = [[]]
    for item in sorted(s, reverse=True):
        for j, free in enumerate(remain):
            if free >= item:
                remain[j] -= item
                soln[j].append(item)
                break
        else:
            soln.append([item])
            remain.append(B-item)

    return soln

def create_gmodel(s,B):
    n = len(s)
    U = len(get_bound(s,B))
    model = Model('cutstock')
    model.Params.OutputFlag = 0
    #setParam('TimeLimit', 25)
    
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
    return model

def csp_colgen_bnb(s,B):
    w,q = [],[]
    for item in sorted(s):
        if w == [] or item != w[-1]:
            w.append(item)
            q.append(1)
        else:
            q[-1] += 1

    # patterns
    t = []
    m = len(w)
    # generate initial patterns
    for i,width in enumerate(w):
        pat = [0]*m
        pat[i] = int(B/width)
        t.append(pat)

    K = len(t)

                
     


s,B = create_instance()
m = create_gmodel(s,B)
bnb = branch_and_bound.gurobi_bnb(m)
print bnb.solve()
