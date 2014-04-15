"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from noncvx_variable import NonCvxVariable
import cvxpy as cp
from cvxpy import settings as s
from collections import OrderedDict

# Use ADMM to attempt non-convex problem.
def admm(self, rho=0.5, iterations=5, solver=cp.ECOS):
    objective,constr_map,dims = self.canonicalize()
    
    new_constr_map = constr_map[s.EQ]
    for c in constr_map[s.INEQ]:
        new_constr_map.add(c)
    vars_ = objective.variables()
    for constr in new_constr_map:
        vars_ += constr.variables()
    var_offsets = OrderedDict()
    vert_offset = 0
    for var in set(vars_):
        var_offsets[var] = vert_offset
        vert_offset += var.size[0]*var.size[1]

    #var_offsets,x_length = self.variables(objective, 
    #                                      new_constr_map)
    
    noncvx_vars = [obj for obj in var_offsets.keys() if isinstance(obj, NonCvxVariable)]
    #import pdb; pdb.set_trace()

    # Form ADMM problem.
    obj = self.objective._expr
    for var in noncvx_vars:
        obj = obj + (rho/2)*sum(cp.square(var - var.z + var.u))
    p = cp.Problem(cp.Minimize(obj), self.constraints)
    # ADMM loop
    for i in range(iterations):
        result = p.solve(solver=solver)
        for var in noncvx_vars:
            var.z.value = var.round(var.value + var.u.value)
            var.u.value = var.value - var.z.value
    # Fix noncvx variables and solve.
    fix_constr = []
    for var in noncvx_vars:
        fix_constr += var.fix(var.z.value)
    p = cp.Problem(self.objective, self.constraints + fix_constr)
    return p.solve(solver=solver)

# Add admm method to cp Problem.
cp.Problem.register_solve("admm", admm)
