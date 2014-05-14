from gurobipy import *
import numpy as np
import pdb

eps = 1e-5

class gurobi_bnb:
    def __init__(self, model):
        self.model = model

    def branch(self):
        model = self.model
        intvars = [iv for iv in model.getVars() if iv.vType != GRB.CONTINUOUS]
        fractional = []
        for v in intvars:
            sol= v.x
            t1= abs(sol - int(sol+0.5))
            if t1> eps:
                fractional.append((v,t1))
        
        if len(fractional) == 0:
            return None
        
        fractional.sort(key=lambda x: x[1])
        return fractional[-1][0]
        
    def bound(self):
        model = self.model
        p1 = model.relax()
        p1.optimize()
        lower_bound = p1.objVal
        if not p1.status == GRB.status.OPTIMAL:
            lower_bound = float('inf')

        p2 = model.relax()
        intvars = [iv for iv in p2.getVars() if iv.vType != GRB.CONTINUOUS]
        for v in intvars:
            sol= v.x
            t1= int(sol+0.5)
            v.lb = t1
            v.ub = t1
        p2.optimize()
        upper_bound = p2.objVal
        if not p2.status == GRB.status.OPTIMAL:
            upper_bound = float('inf')
       
        #pdb.set_trace()
        return {'gap': upper_bound - lower_bound,
                'ub': upper_bound,
                'lb': lower_bound,
                'obj': upper_bound,
                'sol': model.getVars()}


    def _solve(self, i, depth):
        '''
        solve a node and calls solve on children
        '''
        if i> depth:
            return None

        model = self.model
        branch_var = self.branch(model)
        branch_var_name = branch_var.varName
        t1 = branch_var.x

        model_left = model.copy()
        bvar_left = filter(lambda x: x.varName == branch_var_name, model_left.getVars())
        bvar_left.ub = int(t1)
        left_branch = self.bound(model_left)
        
        model_right = model.copy()
        bvar_right = filter(lambda x: x.varName == branch_var_name, model_left.getVars())
        bvar_right.lb = int(t1) + 1 
        right_branch = self.bound(model_right)
      
        picked_left = 0
        if left_branch['obj'] < right_branch['obj']:
            solution = left_branch
            picked_left = 1
        else:
            solution = right_branch

        # update bounds
        solution['lb'] = min(left_branch['lb'], right_branch['lb'])
        solution['ub'] = min(left_branch['ub'], right_branch['ub'])
        solution['gap'] = solution['ub'] -solution['lb']
        
        if solution['gap'] < eps:
            return solution
        
        def take_branch(is_left=1):
            if is_left:
                self._solve(left_branch, i+1, depth)
            else:
                self._solve(right_branch, i+1, depth)

        # take branches
        if left_branch['lb'] < right_branch['lb']:
            left_subtree = take_branch(is_left=1)
            right_subtree = take_branch(is_left=0)
        else:
            right_subtree = take_branch(is_left=0)
            left_subtree = take_branch(is_left=1)

        # propagate solution up the tree
        if left_subtree and right_subtree:
            if left_subtree['obj'] < right_subtree['obj']:
                return left_subtree
            return right_subtree
        if not left_subtree and right_subtree:
            return right_subtree
        if left_subtree and not right_subtree:
            return left_subtree
         
        return solution


    def solve(self, depth=10):
        '''
        model is a gurobi model
        '''
        model = self.model
        res = self.bound()
        if res['gap'] < eps:
            return res['obj']
        return self._solve(model, 0, depth, eps)

