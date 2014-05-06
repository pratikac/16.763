from cvxopt.modeling import variable, op

x,y = variable(), variable()

c1 = (2*x+y <= 3)
c2 = (x+2*y <= 3)
c3 = (x >= 0)
c4 = (y >= 0)

lp1 = op(-4*x-5*y, [c1,c2,c3,c4])
lp1.solve()
