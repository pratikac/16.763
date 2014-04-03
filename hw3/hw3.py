import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb, re
from collections import Counter
from scipy import optimize
import cvxpy as cp
#plt.ion()

airports = ['GDL', 'TIJ','MEX','CUL','REX','SJD','CJS','ACA','MTY']
airports_dict = {airports[i]:airports[i].lower() for i in xrange(len(airports))}
#airports_dict = {airports[i]:i+1 for i in xrange(len(airports))}
flight_rev_dict = {}
flight_dict = {}

def proc_flights():
    global flight_dict, flight_rev_dict
    data_flights = np.array(pd.read_csv('flights.csv'))
    i = 1
    for d in data_flights:
        d[1],d[2] = airports_dict[d[1]], airports_dict[d[2]]
        d[-1] = d[-1] + 30
        flight_rev_dict[d[0]] = i
        i += 1
    for k in flight_rev_dict:
        flight_dict[flight_rev_dict[k]] = k
    for d in data_flights:
        d[0] = flight_rev_dict[d[0]]
    return data_flights

def proc_strings():
    data_strings = np.array(pd.read_csv('routes.csv'))
    data_strings = data_strings[data_strings[:,0]>0]
    for d in data_strings:
        for i in [0, -2, -1]:
            d[i] = int(d[i])
        d[2],d[3] = airports_dict[d[2]], airports_dict[d[3]]
        d[1] = flight_rev_dict[d[1]]
    num_strings = max(data_strings[:,0])

    strings = []
    for i in xrange(1, num_strings+1):
        r = data_strings[data_strings[:,0] == i]
        r = r.tolist()
        for ri in xrange(len(r)-1):
            r[ri][-1] += 30

        strings.append(r)
    return strings

def solve_setcover(flights, strings):
    N = 5
    F = len(flights)
    S = len(strings)
    tc = 333
    
    def get_ground_arcs():
        ground_arcs = []
        for s in strings:
            for i in xrange(len(s)-1):
                ground_arcs.append([s[0][0], s[i][3], s[i][-1], s[i+1][-2]])
        return ground_arcs

    ground_arcs = get_ground_arcs()
    G = len(ground_arcs)

    # 1. prepare a_{is}
    a = np.zeros((F,S))
    for f in xrange(F):
        for s in xrange(S):
            tmp_list = (np.array(strings[s])[:,1]).tolist()
            #print tmp_list
            #pdb.set_trace()
            if (f+1) in tmp_list:
                a[f,s] = 1

    # 2. S_im, S_ip
    Sim = [[] for i in xrange(F)]
    Sip = [[] for i in xrange(F)]
    for i in xrange(F):
        for r in strings:
            if r[0][1] == i:
                Sip[i].append(r[0][0])
            if r[-1][1] == i:
                Sim[i].append(r[0][0])

    #pdb.set_trace()
    def get_num_ac_on_ground_at_time(wt):
        num_ac = 0
        seen_list = set()
        for g in ground_arcs:
            if not g[0] in seen_list:
                if (g[-2] <= wt) and (g[-1] >= wt):
                    num_ac += 1
                seen_list.update([g[0]])
        return num_ac

    # 3. y_{idm}, y_{idp}, y_{iam}, y_{iap}
    yidm,yidp,yiam, yiap = np.zeros((F,)), np.zeros((F,)), \
            np.zeros((F,)),np.zeros((F,))
    for i in xrange(F):
        yidm[i] = get_num_ac_on_ground_at_time(flights[i][-2]-1)
        yidp[i] = get_num_ac_on_ground_at_time(flights[i][-2]+1)

        yiam[i] = get_num_ac_on_ground_at_time(flights[i][-1]-1)
        yiap[i] = get_num_ac_on_ground_at_time(flights[i][-1]+1)
    
    # r_s
    r = np.zeros((S,))
    for ro in strings:
        if (ro[0][-2] <= tc) and (ro[-1][-1] >= tc):
            r[ro[0][0]-1] = 1

    pdb.set_trace()
    # p_g
    p = np.zeros((G,))
    for i in xrange(G):
        g = ground_arcs[i]
        if (g[-2] <= tc) and (g[-1] >= tc):
            p[i] = 1

    # finally, set up the problem!
    x = cp.Variable(S)
    y = cp.Variable(G)
    
    # objective
    f = 0*x[0]
    for s in xrange(S):
        f = f + x[s]

    # c1
    c1 = []
    for i in xrange(F):
        c1.append(sum([a[i,s]*x[s] for s in xrange(S)])-1 == 0)

    # c2
    c2 = []
    for i in xrange(F):
        if len(Sip[i]) > 0:
            c2.append(sum([x[s-1] for s in Sip[i]]) - yidm[i] + yidp[i] == 0)

    # c3
    c3 = []
    for i in xrange(F):
        if len(Sim[i]) > 0:
            c3.append(sum([-x[s-1] for s in Sim[i]]) - yiam[i] + yiap[i] == 0)

    # c4
    c4 = [sum([r[s]*x[s] for s in xrange(S)]) + sum([p[g]*y[g] for g in xrange(G)]) <= N]

    c5 = [y >= 0, x <= 1, 0 <= x]
    
    c6 = [x[0] + x[4] + x[10] + x[11] + x[21] <= 5]

    cons = c1+c2+c3+c4+c5+c6
    #pdb.set_trace()

    prob = cp.Problem(cp.Minimize(f), cons)
    res = prob.solve() 
    print res 

flights = proc_flights()
strings = proc_strings()
solve_setcover(flights, strings)
