"""
The instance of the cutting stock problem is represented by the two
lists of m items of size and quantity s=(s_i) and q=(q_i).

The roll size is B.

Given packing patterns t_1, ...,t_k,...t_K where t_k is a vector of
the numbers of items cut from a roll, the problem is reduced to the
following LP:
    
    minimize    sum_{k} x_k
    subject to  sum_{k} t_k(i) x_k >= q_i    for all i
	            x_k >=0			    for all k.

We apply a column generation approch in
which we generate cutting patterns by solving a
knapsack sub-problem.

"""

from gurobipy import *
import numpy as np
import pdb

LOG = True
EPS = 1.e-6


def solve_column_generation(s,B):
    '''
    solve using column generation
    '''
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

    iter = 0
    K = len(t)
    master = Model("LP")
    
    # add variables
    x = {}
    for k in range(K):
        x[k] = master.addVar(obj=1, vtype="I", name="x[%d]"%k)
    master.update()

    orders={}
    for i in range(m):
        coef = [t[k][i] for k in range(K) if t[k][i] > 0]
        var = [x[k] for k in range(K) if t[k][i] > 0]
        orders[i] = master.addConstr(LinExpr(coef,var), ">", q[i], name="order[%d]"%i)

    master.update()
    master.Params.OutputFlag = 0
    pdb.set_trace()

    while 1:
        iter += 1
        relax = master.relax()
        relax.optimize()
        pi = [c.Pi for c in relax.getConstrs()] # keep dual variables

        knapsack = Model('kp')   # knapsack sub-problem
        knapsack.ModelSense=-1   # maximize
        y = {}
        for i in range(m):
            y[i] = knapsack.addVar(obj=pi[i], ub=q[i], vtype="I", name="y[%d]"%i)
        knapsack.update()

        L = LinExpr(w, [y[i] for i in range(m)])
        knapsack.addConstr(L, "<", B, name="width")
        knapsack.update()
        knapsack.Params.OutputFlag = 0 # silent mode
        knapsack.optimize()
        if LOG:
            print "objective of knapsack problem:", knapsack.ObjVal
        if knapsack.ObjVal < 1+EPS: # break if no more columns
            break

        pat = [int(y[i].X+0.5) for i in y]	# new pattern
        t.append(pat)
        if LOG:
            print "shadow prices and new pattern:"
            for i,d in enumerate(pi):
                print "\t%5d%12g%7d" % (i,d,pat[i])
            print

        # add new column to the master problem
        col = Column()
        for i in range(m):
            if t[-1][i] > 0:
                col.addTerms(t[-1][i], orders[i])
        x[K] = master.addVar(obj=1, vtype="I", name="x[%d]"%K, column=col)
        master.update()
        K += 1

    # Finally, solve the IP
    if LOG:
        master.Params.OutputFlag = 1 # verbose mode
    master.optimize()

    if LOG:
        print 
        print "final solution (integer master problem):  objective =", master.ObjVal
        print "patterns:"
        for k in x:
            if x[k].X > EPS:
                print "pattern", k,
                print "\tsizes:", 
                print [w[i] for i in range(m) if t[k][i]>0 for j in range(t[k][i]) ],
                print "--> %d rolls" % int(x[k].X+.5)

    rolls = []
    for k in x:
        for j in range(int(x[k].X + .5)):
            rolls.append(sorted([w[i] for i in range(m) if t[k][i]>0 for j in range(t[k][i])]))
    rolls.sort()
    return rolls


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

def solve_MIP(s,B):
    '''
        solve the MIP using gurobi
    '''
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

    model.optimize()

    bins = [[] for i in range(U)]
    for (i,j) in x:
        if x[i,j].X > EPS:
            bins[j].append(s[i])
    for i in xrange(bins.count([])):
        bins.remove([])
    for b in bins:
        b.sort()
    bins.sort()
    return bins

def create_example(eg=1, random=False, n=100):
    '''
        B: roll width
        w: orders
        q: quantity of w[i]
    '''
    if not random:
        if eg==1:
            B = 110
            w = [20,45,50,55,75]
            q = [48,35,24,10,8]
            s = []
        else:
            B = 9
            w = [2,3,4,5,6,7,8]
            q = [4,2,6,6,2,2,2]
            s = []
        for j in xrange(len(w)):
            for i in xrange(q[j]):
                s.append(w[j])
        return s,B
    else:
        np.random.seed(3)
        B = 100
        s = [0]*n 
        for i in xrange(n):
            s[i] = np.random.randint(1,99)
        return s,B

if __name__ == "__main__":
    s,B = create_example(random=True, n=200)
    #s,B = create_example(eg=1)

    
    if 1:
        print "\n\n\nColumn generation:"
        rolls = solve_column_generation(s,B)
        print len(rolls), "rolls:"
        print rolls
    else:
        print "\n\n\nMIP:"
        bins = solve_MIP(s,B)
        print len(bins), "bins:"
        print bins
