import pyKriging  
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan
import pickle

import Branin
import numpy as np
import time 

location = 'C:/Users/106/Desktop/shishi.pkl'
k2 = pickle.load(open(location,'rb'))
Branin.huatu2D(k2.predict)
# MSE=Branin.ceshi2D(Branin.AckleyFunction2,k2.predict)
MSE=Branin.ceshi2D(Branin.BraninFunction2,k2.predict)