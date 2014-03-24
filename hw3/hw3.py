import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb, re
from collections import Counter
from scipy import optimize
#plt.ion()

airports = ['GDL', 'TIJ','MEX','CUL','REX','SJD','CJS','ACA','MTY']
airports_dict = {airports[i]:airports[i].lower() for i in xrange(len(airports))}

data = np.array(pd.read_csv('flights.csv'))
for d in data:
    d[1],d[2] = airports_dict[d[1]], airports_dict[d[2]]