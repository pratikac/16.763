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

def draw_roadnet(roadnet):
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

def get_hashable_elist(elist):
    return [(u,v) for u,v,d in elist]

def find_heuristic_P(G, start, end, T, always_select_edges=[], dont_select_edges=[]):
    all_paths = list(nx.all_simple_paths(G, source=start, target=end))
    #print all_paths
    max_paths_to_add = 5
    set_dont_select_edges = set(get_hashable_elist(dont_select_edges))
    set_always_select_edges = set(get_hashable_elist(always_select_edges))
    pruned_paths = []
    num_paths = 0
    for p in all_paths:
        elist = get_edge_list_from_path(G, p)
        hashable_elist = get_hashable_elist(elist)
        # do not select these edges
        if len(set(hashable_elist).intersection(set_dont_select_edges)) > 0:
            continue
        if not set_always_select_edges.issubset(set(hashable_elist)):
            continue

        #print elist
        total_length = sum([d['length'] for u,v,d in elist])
        if total_length <= T:
            if num_paths < max_paths_to_add:
                pruned_paths.append(elist)
                num_paths += 1
    
    #print 'all_paths:', len(all_paths)
    #print 'pruned_paths:', len(pruned_paths)
    return pruned_paths

def print_P(P):
    pi = 0
    for p in P:
        t1 = str(pi) +': '
        for u,v,d in p:
            t1 += str((u,v))
        print t1
        pi += 1

def create_relax_master(P, T):
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

def create_master(G, P, T):
    master,lamda = create_relax_master(P, T)
    
    x = {}
    for u,v,d in G.edges(data=True):
        x[u,v] = master.addVar(obj=0, lb=0, ub=1, vtype='B', name='x[%d,%d]'%(u,v))
    master.update()

    for u,v,d in G.edges(data=True):
        xpij = [1 if (u,v,d) in p else 0 for p in P]
        coef = xpij+[-1]
        var = [lamda[i] for i in xrange(len(P))]
        var += [x[u,v]]
        master.addConstr(LinExpr(coef, var), '=', 0, name='c3[%d,%d]'%(u,v))
    
    return master, lamda,x

def solve_colgen(G, start, end, T):
    P = find_heuristic_P(G, start, end, T)

    while 1:
        relax_master,_ = create_relax_master(P, T)
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
    master, lamda, x = create_master(G, P, T)
    master.optimize()
    print master.objVal
    #print_P(P)

    edges = []
    for u,v,d in G.edges(data=True):
        if x[u,v].x > eps:
            edges.append((u,v))
    print edges


def prune_set_P(P, xij, remove, keep):
    assert remove != keep

    # ugly hack to get variable from string representation
    u = int(xij.split(',')[0].split('[')[-1])
    v = int(xij.split(',')[1].split(']')[0])
    edge = (u,v)

    newP = []
    for p in P:
        helist = get_hashable_elist(p)
        if remove:
            if not edge in helist:
                newP.append(p)
        if keep:
            if edge in helist:
                newP.append(p)
    return newP

def branch(master):
    pass

def bound(model, G, P, T):
   pass 
    
def _solve_colgen(model, G, P, T, i, depth):
    '''
    solves the node and calls solve on its children
    '''
    if i > depth:
        return None

    bvar = branch_colgen(model)
    bvar_name = bvar.varName
    t1 = bvar.x

    model_left = model.copy()
    bvar_left = filter(lambda z: z.varname == bvar_name, model_left.getVars())
    bvar_left.ub = int(t1)
    #P_left = prune_set_P(P, bvar_name, remove=1)
    branch_left = bound(model_left, G, T)

    model_right = model.copy()
    bvar_right = filter(lambda z: z.varname == bvar_name, model_right.getVars())
    bvar_right.lb = int(t1)+1
    #P_right = prune_set_P(P, bvar_name, keep=1)
    branch_right = bound(model_right, G, T)

    # decide solution
    picked_left = 0
    if branch_left['obj'] < branch_right['obj']:
        solution = branch_left
    else:
        solution = branch_right

    # update bounds
    solution['lb'] = min(branch_left['lb'], branch_right['lb'])
    solution['ub'] = min(branch_left['ub'], branch_right['ub'])
    solution['gap'] = solution['ub'] - solution['lb']

    if solution['gap'] < eps:
        return solution
    
    def take_branch(is_left = 1):
        if is_left:
            _solve_colgen(model_left, G, P, T, i+1, depth)
        else:
            _solve_colgen(model_right, G, P, T, i+1, depth)

    # take branches
    if branch_left['lb'] < branch_right['ub']:
        subtree_left = take_branch(is_left=1)
        subtree_right = take_branch(is_left = 0)
    else:
        subtree_right = take_branch(is_left = 0)
        subtree_left = take_branch(is_left=1)

    # propagate bounds
    if subtree_left and subtree_right:
        if subtree_left['obj'] < subtree_right['obj']:
            return subtree_left
        return subtree_right
    if not subtree_left and subtree_right:
        return subtree_right
    if subtree_left and not subtree_right:
        return subtree_left
    
    # if reach depth, return solution
    return solution


def solve_colgen_bnb(G, start, end, T, depth=10):
    
    P = find_heuristic_P(G, start, end,T)
    master,lamda,x = create_master(G, P, T)
    res = bound(master, G, P, T)
    if res['gap'] < eps:
        return res['obj'], res['solution']
    return _solve_colgen(master, G, P, T, 0, depth)

    #while 1:
    #    relax_master,_ = create_master


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
#draw_roadnet(G)
