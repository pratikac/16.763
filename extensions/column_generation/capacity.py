import networkx as nx
#from gurobipy import *
import numpy as np
import random, itertools
import pylab as plt
import pdb


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

def find_heuristic_P(G, start, end, T):
    prob_pick = 0.3
    def get_edge_list_from_path(p):
        elist = []
        for vi in xrange(len(p)-1):
            e = filter(lambda x: x[1] == p[vi+1], G.edges(p[vi],data=True))[0]      # this has to be unique
            elist.append(e)
        return elist

    all_paths = list(nx.all_simple_paths(G, source=start, target=end))
    #print all_paths
    pruned_paths = []
    for p in all_paths:
        elist = get_edge_list_from_path(p)
        #print elist
        total_length = sum([d['length'] for u,v,d in elist])
        if total_length <= T:
            if np.random.random_sample() < prob_pick:
                pruned_paths.append(p)
     
    print 'all_paths:', all_paths
    print 'pruned_paths:', pruned_paths

def solve_colgen(G, start, end, T):
    pass

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

    model.optimize()

    path = []
    for i,j in x:
        if x[i,j].X > EPS:
            path.append((i,j))
    return path

G = create_eg_roadnet()
#G = create_road_network()
find_heuristic_P(G, 0, len(G)-1, 14)
test_roadnet(G)
#print find_all_paths(G,0,len(G)-1)

#solve_mip(G)

