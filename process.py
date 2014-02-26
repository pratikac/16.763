import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb, re
from collections import Counter

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
oper_dict = {'VMC':0,
'IMC':1
}

def proc_data():
    data = np.array(pd.read_csv('table1.csv'))
    data = data[:,2:]
    #pdb.set_trace()
    for i in xrange(len(data)):
        data[i,1] = 4*data[i,0] + data[i,1]
        data[i,2] = oper_dict[data[i,2]]
        data[i,3] = config_dict[data[i,3]]
    return data[:,1:]

def convert_to_quater(s):
    s = re.split(' |:', s)
    hr, m = int(s[0]), int(s[1])
    if s[2] == 'PM':
        hr += 12
    return ((hr*60+m), int((hr*60 + m)/15))

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

'''
todo:
    average by time of day
'''
def filter_data(data, oper, config):
    t1 = data[data[:,1] == oper_dict[oper],:]
    t1 = data[data[:,2] == config_dict[config],:]
    t1 = t1[:,3:5]
    return t1

def plot_capacity(t1key, t1count, oper, config):
    t1 = np.asarray(t1key)
    fig = plt.figure(1)
    for i in xrange(len(t1)):
        plt.plot(t1[i,0], t1[i,1], 'o',
                ms=4*t1count[i], markerfacecolor=None,
                markeredgecolor='blue',
                fillstyle='none',
                markeredgewidth=1.5)
    plt.xlim([0,20])
    plt.ylim([0,20])
    plt.xlabel('departures')
    plt.ylabel('arrivals')
    plt.title('oper: ' + oper+' , config: ' + config)
    #plt.show()
    
    plt.savefig(filename_dict[oper]+'_'+filename_dict[config]+'.png')

def plot_data(data):
    for oper in oper_dict.keys():
        for config in config_dict.keys():
            print oper, config
            if oper == 'VMC' and config == '28L, 28R | 28L, 28R':
                t1 = filter_data(data, oper, config)
                t1s = t1.tolist()
                t1s = [tuple(i) for i in t1s]
                t1dict = dict(Counter(t1s))
                pdb.set_trace()
                #plot_capacity(t1dict.keys(), t1dict.values(), oper, config)        
                        
if __name__=="__main__":
    data = proc_data()
    arr = proc_arr()
    dep = proc_dep()
    plot_data(data)
