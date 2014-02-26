import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb, re
from collections import Counter
from scipy import optimize
#plt.ion()

filename_dict = {'28L, 28R | 28L, 28R':'28l_28r_n_28l_28r',
'28R | 1R':'28r_n_1r',
'VMC':'vmc',
'IMC':'imc',
'28R | 28L, 28R':'28r_n_28l_28r',
'28L, 28R | 1L, 1R':'28l_28r_n_1l_1r',
'28R | 1L, 1R':'28r_n_1l_1r',
'28L | 1L, 1R':'28l_n_1l_1r',
'28L | 28L':'28l_n_28l'
}

config_dict = {'28L, 28R | 28L, 28R':0,
'28R | 1R':1,
'28R | 28L, 28R':2,
'28L, 28R | 1L, 1R':3,
'28R | 1L, 1R':4,
'28L | 1L, 1R':5,
'28L | 28L':6
}
oper_dict = {'VMC':7,
'IMC':8
}

def proc_data():
    data = np.array(pd.read_csv('table1.csv'))
    data = data[:,2:]
    for i in xrange(len(data)):
        data[i,1] = 4*data[i,0] + data[i,1]
        data[i,2] = oper_dict[data[i,2]]
        data[i,3] = config_dict[data[i,3]]
    t1 = data[:,4].copy()
    data[:,4] = data[:,5].copy()
    data[:,5] = t1
    return data[:,1:-1]

def convert_to_quater(s):
    s = re.split(' |:', s)
    hr, m = int(s[0]), int(s[1])
    if hr == 12:
        hr = 0
    if s[2] == 'PM':
        hr += 12
    return ((hr*60+m), int((hr*60 + m)/15)+1)

def proc_arr():
    arr = np.array(pd.read_csv('arrivals.csv'))
    arr = arr[1:,3:5]
    arr2 = []
    for i in xrange(len(arr)):
        if isinstance(arr[i,0],str) and isinstance(arr[i,1], str):
            sched_m, sched_q = convert_to_quater(arr[i,0])
            actual_m, actual_q = convert_to_quater(arr[i,1])
            arr2.append([sched_m, sched_q, actual_m, actual_q])
    #pdb.set_trace()
    return np.array(arr2)

def proc_dep():
    dep = np.array(pd.read_csv('departures.csv'))
    dep = dep[1:,3:5]
    dep2 = []
    for i in xrange(len(dep)):
        if isinstance(dep[i,0],str) and isinstance(dep[i,1], str):
            sched_m, sched_q = convert_to_quater(dep[i,0])
            actual_m, actual_q = convert_to_quater(dep[i,1])
            dep2.append([sched_m, sched_q, actual_m, actual_q])
    #pdb.set_trace()
    return np.array(dep2)

# create new dataset from arr, dep
def correlate_data(arr, dep):
    arr, dep = arr.tolist(), dep.tolist()
    datadic = {}
    contdic = {}
    for a in arr:
        is_busy = a[2] > a[0]
        if a[1] < 4:
            is_busy = False
        is_busy = int(is_busy)
        if not a[3] in datadic:
            datadic[a[3]] = [-1, -1, 1, 0]
            contdic[a[3]] = [is_busy, 0]
        else:
            datadic[a[3]][2] += 1
            contdic[a[3]][0] = contdic[a[3]][0] or is_busy

    for d in dep:
        is_busy = d[2] > d[0]
        if d[1] > 92:
            is_busy = False
        is_busy = int(is_busy)
        if not d[3] in datadic:
            datadic[d[3]] = [-1, -1, 0, 1]
            contdic[d[3]] = [0, is_busy]
        else:
            datadic[d[3]][3] += 1
            contdic[d[3]][1] = contdic[d[3]][1] or is_busy
    
    data = []
    for k in datadic:
        data.append([k]+datadic[k])
    return np.array(data)


'''
todo:
    average by time of day
'''
def filter_data(data, oper, config):
    #pdb.set_trace()
    t1 = data[data[:,1] == oper_dict[oper]].copy()
    t2 = t1[t1[:,2] == config_dict[config]]
    t2 = t2[:,3:5]
    return t2

def plot_capacity(t1, oper, config):
    t1c = list(t1.tolist())
    t1s = t1c
    t1s = [tuple(i) for i in t1s]
    t1dict = dict(Counter(t1s))
    t1keys, t1count = t1dict.keys(), t1dict.values()
    #pdb.set_trace()
    plt.figure()
    for i in xrange(len(t1keys)):
        plt.plot(t1keys[i][0], t1keys[i][1], 'o', ms=8),
                #ms=5*t1count[i], markerfacecolor=None,
                #markeredgecolor='blue',
                #fillstyle='none',
                #markeredgewidth=1.5)
    
    plt.xlim([0,25])
    plt.ylim([0,25])
    plt.xlabel('arrivals')
    plt.ylabel('departures')
    plt.grid()
    plt.title('oper: ' + oper+' , config: ' + config)
    fname = filename_dict[oper]+'_'+filename_dict[config]+'.png'
    print 'Saving: ', fname
    plt.savefig(fname)

def plot_all(points, capacities, alg, labels):
    colors = 'bgrcmykw'
    plt.figure()
    plt.plot(points[:,0], points[:,1], 'o', ms=7, label='', alpha=0.3)
    for ci in xrange(len(capacities)):
        c = capacities[ci]
        plt.plot(c[:,0], c[:,1], '-o', lw=3, ms=3, markerfacecolor=plt.cm.jet(ci),
                label=labels[ci])
    
    plt.xlim([0,25])
    plt.ylim([0,25])
    plt.xlabel('arrivals')
    plt.ylabel('departures')
    plt.grid()
    plt.legend(prop={'size':12})
    fname = alg+'.pdf'
    print 'Saving: ', fname
    plt.savefig(fname)

def run_opt(t1, oper, config, which):
    if(which == 'lsq'):
        return least_squares_regression(t1)
    elif(which == 'cvx'):
        return convex_hull(t1)
    elif(which == 'cvx2'):
        return convex_hull2(t1)
    elif(which == 'plt'):
        return plot_capacity(t1, oper, config)        

def crunch_data(data, which_oper, which, overall_flag=0):
    if overall_flag:
        t1 = data[:,3:5]
        run_opt(t1, _, _, which)
    else:
        '''
        test_tuples = [
                ('VMC', '28L, 28R | 1L, 1R'),
                ('VMC', '28L, 28R | 28L, 28R'),
                ('VMC', '28R | 1R'),
                ('IMC', '28L, 28R | 1L, 1R'),
                ('IMC', '28R | 1L, 1R'),
                ('IMC', '28L | 1L, 1R')]
        '''
        test_tuples = [
                ('VMC', '28L, 28R | 28L, 28R')]
        #test_tuples = [(which_oper, x) for x in config_dict]
        capacities = []
        labels = []
        for test_tup in test_tuples:
            oper, config = test_tup
            #print oper, config
            t1 = filter_data(data, oper, config)
            if len(t1) == 0:
                continue
            c = run_opt(t1, oper, config, which)
            capacities.append(c)
            labels.append(oper+' '+config)

        if not which == 'plt':
            plot_all(data[:,3:5], capacities, which_oper+'_'+which, labels)

def convex_hull(points):
    m = np.max(points[:,0])
    N = len(points)
    # N z and m: alpha, beta
    def f(x):
        return np.sum(x[:N])
    def ieqcons(x):
        ieqcons = []
        # 1. slope constraint
        for i in xrange(m-1):
            ieqcons.append(x[N+m+i] - x[N+m+i+1])
        # 2. -beta_1 >= 0
        ieqcons.append(-x[N+m])
        
        # 3. z_n > (d-dtilde)
        dtilde = np.array([x[max(0,N+a-1)] + x[max(N+m,N+m+a-1)]*a for a in points[:,0]])
        ieqcons = ieqcons + (x[:N] - (points[:,1] - dtilde)).tolist()
        
        # 4. z_n > 0
        for i in xrange(N):
            ieqcons.append(x[i])
        
        return ieqcons

    def eqcons(x):
        eqcons = []
        # 3. continuity
        for i in xrange(m-1):
            eqcons.append(x[N+i]+x[N+m+i]*i - x[N+i+1]-x[N+m+i+1]*i)
        return np.array(eqcons)
    
    xmin = optimize.fmin_slsqp(f, np.zeros((N+2*m,1)),
            f_eqcons = eqcons, f_ieqcons=ieqcons, acc=1e-6)
    
    print xmin[N:]
    toplot = [[i, xmin[max(0,N+i-1)] + xmin[max(N+m,N+m+i-1)]*i] for i in 
            np.arange(1,m)]
    md = np.max(points[points[:,0]==m,1])
    toplot.append([m,md])
    toplot.append([m,0])
    return np.array(toplot)

def convex_hull2(points):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    toplot = []
    for simplex in hull.simplices:
        toplot.append([points[simplex,0], points[simplex,1]])
    return np.array(toplot)

def least_squares_regression(points):
    m = np.max(points[:,0])+1
    # m alpha, beta
    def f(x):
        #pdb.set_trace()
        dtilde = np.array([x[max(0,a-1)] + x[max(m,m+a-1)]*a for a in points[:,0]])
        return np.linalg.norm(points[:,1] - dtilde)**2
    def ieqcons(x):
        ieqcons = []
        # 1. slope constraint
        for i in xrange(m-1):
            ieqcons.append(x[m+i] - x[m+i+1])
        # 2. -beta_1 >= 0
        ieqcons.append(-x[m]-0.1)
        #print ieqcons
        return np.array(ieqcons)
    def eqcons(x):
        eqcons = []
        # 3. continuity
        for i in xrange(m-1):
            eqcons.append(x[i]+x[m+i]*i - x[i+1]-x[m+i+1]*i)
        return np.array(eqcons)
    
    xmin = optimize.fmin_slsqp(f, np.random.random((2*m,1)),
            f_eqcons = eqcons, f_ieqcons=ieqcons)

    toplot = np.array([[i, xmin[max(0,i-1)] + xmin[max(m,m+i-1)]*i] for i in 
            np.arange(1,m)])
    return toplot

def test():
    def f(x):
        return np.sqrt((x[0]-3)**2 + (x[1]-2)**2)
    def cons(x):
        return np.array([x[0] -2, x[1] - 3])

    xmin = optimize.fmin_slsqp(f, np.array([0,0]), f_ieqcons=cons)
    print xmin

def main():
    data = proc_data()
    arr = proc_arr()
    dep = proc_dep()
    new_data = correlate_data(arr, dep)
    '''
    for oper in ['VMC', 'IMC']:
        for alg in ['lsq', 'cvx']:
            crunch_data(data, oper, alg, 0)
    '''
    crunch_data(data, 'VMC', 'cvx2', 0)
    #crunch_data(new_data, 'cvx', 1)

if __name__=="__main__":
    main()
