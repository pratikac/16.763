from numpy import *
from pylab import *


rc('font', family='serif')
rc('text', usetex='True')

colgen = [0.061720848083496094, 0.1344590187072754, 0.2555210590362549, 0.3769710063934326, 0.7768020629882812, 0.6432981491088867, 0.9290788173675537, 1.153012990951538, 1.9432380199432373]
mip = [0.005011796951293945, 0.019346952438354492, 0.06307196617126465, 0.13258695602416992, 1.641205072402954, 25.138794898986816, 25.107528924942017, 25.12, 25.16562008857727]

N = np.arange(10,100,10)

fig = figure(1)
ax = fig.add_subplot(111)
setp(ax.get_xticklabels(), fontsize=20)
setp(ax.get_yticklabels(), fontsize=20)

plot(N, colgen, 'bo--', lw=2, ms=8, label=r'Column generation')
plot(N, mip, 'ro--', lw=2, ms=8, label=r'MIP formulation')

xlabel(r'N (size of the problem)')
ylabel(r'Time (s)')
legend()
savefig('colgen_results.pdf', bbox_inches='tight')
show()