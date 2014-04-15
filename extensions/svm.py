import cvxpy as cp
import mixed_integer as mi
import cvxopt

N = 50
M = 40
n = 10
data = []
for i in range(N):
    data += [(1, cvxopt.normal(n, mean=1.0, std=2.0))]
for i in range(M):
    data += [(-1, cvxopt.normal(n, mean=-1.0, std=2.0))]

# Construct problem.
gamma = cp.Parameter(sign="positive")
gamma.value = 0.1
# 'a' is a variable constrained to have at most 6 non-zero entries.
#a = mi.SparseVar(n, nonzeros=6)
a = cp.Variable(n)
b = cp.Variable()

slack = [cp.pos(1 - label*(sample.T*a - b)) for (label, sample) in data]
objective = cp.Minimize(cp.norm(a, 2) + gamma*sum(slack))
p = cp.Problem(objective)
# Extensions can attach new solve methods to the CVXPY Problem class.
p.solve(method="admm")

# Count misclassifications.
errors = 0
for label, sample in data:
    if label*(sample.T*a - b).value < 0:
        errors += 1

print "%s misclassifications" % errors
print a.value
print b.value
