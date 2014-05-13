from gurobipy import *
import numpy as np


def branch_and_bound(prob):
    '''
    prob is a gurobi model
    '''
    lb = prob.relax()
    
