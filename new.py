import csv
import time
import xlwt
import pickle
import numpy as np
from scipy.stats import spearmanr
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets.supervised import SupervisedDataSet as SDS
test_file = 'data/data2.csv'
model_file = 'model.pkl'
t1=time.time()
# load model
c=[]
b=[]
a=[]
net = pickle.load( open( model_file, 'rb' ))
test = np.loadtxt( test_file, delimiter = ';' )

x_test = test[:,0:-1]
y_test = test[:,-1]
y_test = y_test.reshape( -1, 1 )
input_size = x_test.shape[1]
target_size = y_test.shape[1]

# prepare dataset

ds = SDS( input_size, target_size )
ds.setField( 'input', x_test )
ds.setField( 'target', y_test )
p = net.activateOnDataset(ds)
for row in ds:
     a.append(float(row[-1]))
for row in p:
     b.append(float(row[-1]))


print(spearmanr(a,b))


print(time.time()-t1)