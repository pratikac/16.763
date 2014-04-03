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
        strings.append(r.tolist())
    return strings

def solve_setcover(flights, strings):
    N = 5
    F = len(flights)
    S = len(strings)
    tc = 333
    G = 0
    for r in strings:
        G += (len(r) -1)

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

    # 3. y_{idm}, y_{idp}, y_{iam}, y_{iap}
    yidm,yidp,yiam, yiap = np.zeros((F,)), np.zeros((F,)), \
            np.zeros((F,)),np.zeros((F,))


    # r_s
    r = np.zeros((S,))
    for ro in strings:
        if (ro[0][-2] <= tc) and (ro[-1][-1] >= tc):
            r[ro[0][0]-1] = 1

    def get_ground_arcs():
        ground_arcs = []
        for s in strings:
            for i in xrange(len(s)-1):
                ground_arcs.append([s[0][0], s[i][3], s[i][-1], s[i+1][-2]])
        pdb.set_trace()

    ground_arcs = get_ground_arcs()

    x = cp.Variable(S)
    y = cp.Variable(G)

flights = proc_flights()
strings = proc_strings()
solve_setcover(flights, strings)
