import numpy as np
from gurobipy import *
#import pandas as pd
import pdb

cap = [150, 125, 150, 125]
odif = {}
demand, fares, legs = {}, {}, {}

def read_data():
    global demand, fares, legs
    global od, classes

    rmdata = np.loadtxt('denhub.csv')
    #rmdata = np.array(pd.read_csv('denhub.csv'))
    #rmdata = np.array(pd.read_csv('denhublcc.csv'))

    for r in rmdata:
        t1 = (r[1],r[2],r[3])
        if not t1 in odif.keys():
            odif[t1] = 1

        demand[t1] = r[4]
        fares[t1] = r[5]

        t3 = r[0].split(' ')
        t3 = [int(t3i) for t3i in t3]
        for t3i in t3:
            if not t3i in legs:
                legs[t3i] = []
            legs[t3i].append(t1)

    return

def solve_netrm():
    m = Model('netrm')
    x = {}
    for i,j,k in odif.keys():
        x[i,j,k] = m.addVar(vtype='I', ub=demand[i,j,k], obj=-fares[i,j,k],
                            name='x[%s,%s,%s]' %(i,j,k))
    m.update()

    # leg capacity
    for l in legs.keys():
        m.addConstr(
            quicksum(x[xi] for xi in legs[l]) <= cap[l-1],
                    'leg[%s]'%(l))

    m.optimize()

    # print sol
    #pdb.set_trace()
    if m.status == GRB.status.OPTIMAL:
        #sol = m.getAttr('x',x)
        for i,j,k in odif.keys():
            print('%s & %s & %s & %s & %s') %(i,j,k,x[i,j,k].x,demand[i,j,k])

        #pdb.set_trace()
        #  get dual variables
        relax = m.relax()
        relax.optimize()
        pi = [c.Pi for c in relax.getConstrs()]
        print pi
        #for i,d in enumerate(pi):
        #    print

read_data()
solve_netrm()
