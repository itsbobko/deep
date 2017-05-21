"train a regression MLP"
import csv
import numpy as np
import pickle
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

train_file = 'data/data2.csv'
validation_file = 'data/test.csv'
output_model_file = 'model.pkl'

hidden_size = 600
epochs = 200

# load data
train = np.loadtxt( train_file, delimiter = ';' )

validation = np.loadtxt( validation_file, delimiter = ';' )

x_train = train[:,0:-1]

y_train = train[:,-1]
y_train = y_train.reshape( -1, 1 )
print(y_train)
input_size = x_train.shape[1]

target_size = y_train.shape[1]

# prepare dataset
hidden_size=round(1.5*input_size)
hidden_size2=round(3*input_size)
ds = SDS( input_size, target_size )
ds.setField( 'input', x_train )
ds.setField( 'target', y_train )

# init and train

net = buildNetwork( input_size, hidden_size,hidden_size2, target_size)#, bias=True )
#net = buildNetwork( input_size, hidden_size, target_size)
trainer = BackpropTrainer( net,ds )

print ("training for {} epochs...".format( epochs ))

for i in range( epochs ):
	mse = trainer.train()
	rmse = sqrt( mse )
	print ("training RMSE, epoch {}: {}".format( i + 1, rmse ))
	
pickle.dump( net, open( output_model_file, 'wb' ))






