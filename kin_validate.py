"train/validate to find out how many epochs to train"

import numpy as np
import pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

train_file = 'data/data2.csv'
validation_file = 'data/test.csv'
output_model_file = 'model_val.pkl'

hidden_size = 600
epochs = 1000
continue_epochs = 10	
validation_proportion = 0.15

# http://fastml.com/pybrain-a-simple-neural-networks-library-in-python/

train = np.loadtxt( train_file, delimiter = ';' )
validation = np.loadtxt( validation_file, delimiter = ';' )
#train = np.vstack(( train, validation ))

x_train = train[:,0:-1]
y_train = train[:,-1]
y_train = y_train.reshape( -1, 1 )

input_size = x_train.shape[1]
target_size = y_train.shape[1]
hidden_size=round(1.5*input_size)
hidden_size2=round(3*input_size)
# prepare dataset

ds = SDS( input_size, target_size )
ds.setField( 'input', x_train )
ds.setField( 'target', y_train )

# init and train

net = buildNetwork( input_size, hidden_size,hidden_size2, target_size, bias=True )
trainer = BackpropTrainer( net,ds )

train_mse, validation_mse = trainer.trainUntilConvergence( verbose = True, validationProportion = validation_proportion, 
	maxEpochs = epochs, continueEpochs = continue_epochs )
print(train_mse)
print(validation_mse)
pickle.dump( net, open( output_model_file, 'wb' ))






