import numpy as np
from gurobipy import *
import pandas as pd


airports= {'DEN':0, 'SFO':1, 'MIA':2, 'LAX':3, 'ORD':4}
classes={'Y':0,'B':1,'M':2}
cap = [150, 125, 150, 125]

def read_data():
    demand, fares = np.zeros((5,5,3)), np.zeros((5,5,3))
    leg_map = [[],[],[]]
    rmdata = np.array(pd.read_csv('denhub.csv'))
    for r in rmdata:
        r[1], r[2] = airports[r[1]], airports[r[2]]
        r[3] = classes[r[3]]
        fares[r[1],r[2],r[3]] = r[-1]
        demand[r[1],r[2],r[3]] = r[-2]

        legs = r[0].split(' ')
        for l in legs:
            leg_map[int(l)-1].append([r[1],r[2],r[3]])

    return demand, fares, leg_map


def solve_netrm():
    demand, fares, leg_map = read_data()
    m = Model('netrm')
    x = m.addVar(vtype=GRB.INTEGER, name='x')
    m.update()

    x = mi.IntVar((5,5,3))
    f = 0.x[0,0,0]
    f = sum([sum([ sum([x[i,j,k] for k in xrange(3)]) for j in xrange(5)]) \
            for i in xrange(5)])
    import pdb; pdb.set_trace()


solve_netrm()
