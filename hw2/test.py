import cvxpy as cp
import cvxopt

# Problem data.
m = 30
n = 20
A = cvxopt.normal(m,n)
b = cvxopt.normal(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(sum(cp.square(A*x - b)))
constraints = [0 <= x, x <= 1]
p = cp.Problem(objective, constraints)

# The optimal objective is returned by p.solve().
result = p.solve()
# The optimal value for x is stored in x.value.
print x.value
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
#print constraints[0].dual_value