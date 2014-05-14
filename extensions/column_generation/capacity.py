import networkx as nx
from gurobipy import *
import numpy as np
import random, itertools
import pylab as plt
import pdb

eps = 1e-5

def create_road_network(N=10, mu=20.):
    g = nx.erdos_renyi_graph(N, 0.51)
    g = nx.connected_component_subgraphs(g)[0]

    def roadmaker():
        for i in itertools.count():
            yield 'road%d' %i, int(np.random.exponential(mu)), int(10 + 10*np.random.random_sample())
    road_iter = roadmaker()

    roadnet = nx.MultiDiGraph()
    for i,(u,v,data) in enumerate(g.edges_iter(data=True)):
        label,length,capacity = road_iter.next()
        roadnet.add_edge(u,v,label,length=length,capacity=capacity)

    return roadnet

def test_roadnet(roadnet):
    pos = nx.spring_layout(roadnet)

    edge_labels = dict([ ((u,v,), (d['capacity'], d['length']))
                        for u,v,d in roadnet.edges(data=True)])

    plt.figure(1)
    nx.draw(roadnet, pos)
    nx.draw_networkx_edge_labels(roadnet,pos, edge_labels=edge_labels)
    plt.show()

def create_eg_roadnet():
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(6))
    
    edges = [   (1,2,1,10),
                (2,4,1,1),
                (1,3,10,3),
                (3,2,1,2),
                (2,5,2,3),
                (3,4,5,7),
                (3,5,12,3),
                (4,5,10,1),
                (5,6,2,2),
                (4,6,1,7)
    ]
    for e in edges:
        G.add_edge(e[0]-1,e[1]-1,label='e[%d,%d]'%(e[0]-1,e[1]-1), length=e[3],capacity=e[2])
    return G

def get_edge_list_from_path(G, p):
    elist = []
    for vi in xrange(len(p)-1):
        e = filter(lambda x: x[1] == p[vi+1], G.edges(p[vi],data=True))[0]      # this has to be unique
        elist.append(e)
    return elist

def find_heuristic_P(G, start, end, T):
    all_paths = list(nx.all_simple_paths(G, source=start, target=end))
    max_paths_to_add = len(all_paths)/4
    #print all_paths
    pruned_paths = []
    num_paths = 0
    for p in all_paths:
        elist = get_edge_list_from_path(G, p)
        #print elist
        total_length = sum([d['length'] for u,v,d in elist])
        if total_length <= T:
            if num_paths < max_paths_to_add:
                pruned_paths.append(elist)
                num_paths += 1
     
    #print 'all_paths:', len(all_paths)
    #print 'pruned_paths:', len(pruned_paths)
    return pruned_paths

def solve_colgen(G, start, end, T):
    P = find_heuristic_P(G, start, end, T)
    #pdb.set_trace()

    def print_P(P):
        pi = 0
        for p in P:
            t1 = str(pi) +': '
            for u,v,d in p:
                t1 += str((u,v))
            print t1
            pi += 1

    def create_master(P):
        relax_master = Model('master')
        lamda={}
        pi = 0
        for p in P:
            sum_cij = 0
            for u,v,d in p:
                sum_cij += d['capacity']
            lamda[pi] = relax_master.addVar(obj=sum_cij, lb = 0, vtype=GRB.CONTINUOUS, name='lamda[%d]'%pi)
            pi += 1
        relax_master.update()
        
        # constraints
        # resource
        pi = 0
        coef = [1]*len(P)
        var = [lamda[i] for i in xrange(len(P))]
        for p in P:
            sum_tij = 0
            for u,v,d in p:
                sum_tij += d['length']
            coef[pi] = sum_tij
            pi += 1
        relax_master.addConstr(LinExpr(coef, var), '<=', T, name='c1')

        # sum lamba = 1
        coef = [1]*len(P)
        relax_master.addConstr(LinExpr(coef, var), '=', 1, name='c2')

        relax_master.update()
        #relax_master.write('cap.lp')
        return relax_master, lamda

    while 1:
        relax_master,_ = create_master(P)
        relax_master.optimize()
        duals = [c.Pi for c in relax_master.getConstrs()]

        Gc = G.copy()
        for u,v,d in Gc.edges(data=True):
            d['capacity'] = d['capacity'] - duals[0]*d['length']
        
        len_sc = nx.shortest_path_length(Gc, source=start, target=end, weight='capacity')
        if len_sc - duals[1] >= 0:
            print 'Found optimal solution'
            #print optimal solution here
            break
        else:
            new_path = nx.shortest_path(Gc, source=start, target=end, weight='capacity')
            new_path_elist = get_edge_list_from_path(G, new_path)
            P.append(new_path_elist)
            print 'added new path: ', new_path

    #pdb.set_trace()
    # use the integrality constraints now
    master, lamda = create_master(P)
    x = {}
    for u,v,d in G.edges(data=True):
        x[u,v] = master.addVar(obj=0, vtype='B', name='x[%d,%d]'%(u,v))
    master.update()

    for u,v,d in G.edges(data=True):
        xpij = [1 if (u,v,d) in p else 0 for p in P]
        coef = xpij+[-1]
        var = [lamda[i] for i in xrange(len(P))]
        var += [x[u,v]]
        master.addConstr(LinExpr(coef, var), '=', 0, name='c3[%d,%d]'%(u,v))
    
    master.optimize()
    print master.objVal
    print_P(P)

    edges = []
    for u,v,d in G.edges(data=True):
        if x[u,v].x > eps:
            edges.append((u,v))
    print edges

def solve_mip(G, start, end, T):
    model = Model('G_mip')
    setParam('TimeLimit', 50)
    model.Params.OutputFlag = 1

    # 1. add vars
    x = {}
    for u,v,d in G.edges(data=True):
        x[u,v] = model.addVar(obj=d['capacity'], vtype='B', name='x[%d,%d]'%(u,v))
        #print u,v,d['length'],d['capacity']
    model.update()

    # 2. constraints
    # (a)
    start_edges = [x[u,v] for u,v in G.edges(data=False) if u==start]
    coeffs = [1 for e in start_edges]
    model.addConstr(LinExpr(coeffs, start_edges), '=', 1, name='c1')

    # (b)
    for i in xrange(len(G)):
        if (not i == start) and (not i == end):
            lhs = [x[u,j] for u,j in G.edges(data=False) if i ==u]
            rhs = [x[j,u] for j,u in G.edges(data=False) if u ==i]
            coeffs = [1 for e in lhs]+[-1 for e in rhs]
            model.addConstr(LinExpr(coeffs, lhs+rhs), '=', 0, name='c2[%d]'%i)

    # (c)
    end_edges = [x[u,v] for u,v in G.edges(data=False) if v==end]
    coeffs = [1 for e in end_edges]
    model.addConstr(LinExpr(coeffs, end_edges), '=', 1, name='c3')

    # (d)
    terms = [d['length']*x[i,j] for i,j,d in G.edges(data=True)]
    model.addConstr(quicksum(terms), '<=', T, name='c4')
    
    #model.update()
    #model.write('cap.lp')
    model.optimize()

    path = []
    for i,j in x:
        if x[i,j].X > eps:
            path.append((i,j))
    return path

G = create_eg_roadnet()
#G = create_road_network()
#find_heuristic_P(G, 0, len(G)-1, 14)
#print find_all_paths(G,0,len(G)-1)

#print solve_mip(G, 0, 5, 14)
solve_colgen(G, 0, 5, 13)
test_roadnet(G)
