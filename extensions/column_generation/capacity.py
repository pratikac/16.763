import networkx as nx
from gurobipy import *
import numpy as np
import random, itertools


def create_road_network(N=10, mu=1.):
    g = nx.erdos_renyi_graph(N, 0.3)
    g = nx.connected_component_subgraphs(g)[0]

    def roadmaker():
        for i in itertools.count():
            yield 'road%d' %i, np.random.exponential(mu), np.random.random_sample()
    road_iter = roadmaker()

    roadnet = nx.MultiDiGraph()
    for i,(u,v,data) in enumerate(g.edges_iter(data=True)):
        label,length,capacity = road_iter.next()
        roadnet.add_edge(u,v,label,length=length,capacity=capacity)

    return roadnet
