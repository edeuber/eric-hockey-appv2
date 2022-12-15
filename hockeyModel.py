# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:20:53 2022

@author: deube
"""

import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import tensorflow as tf

#Extract data from csv
raw_csv_data = np.loadtxt('CombinedDataModelNoHeading.csv',delimiter=',')


unscaled_inputs_all = raw_csv_data[:,0:-1]

targets_all = raw_csv_data[:,-1]


##We do not need to balance the dataset because the probability of certain outcomes is inherently less likely
##than other outcomes

#Standardize the inputs
#scaled_inputs = preprocessing.scale(unscaled_inputs_all)

##################
#Shuffle the data#
##################

shuffled_indices = np.arange(unscaled_inputs_all.shape[0])
np.random.shuffle(shuffled_indices)



# Use the shuffled indices to shuffle the inputs and targets.
shuffled_inputs = unscaled_inputs_all[shuffled_indices]
shuffled_targets = targets_all[shuffled_indices]

################
#Split the data#
################

# Count the total number of samples
samples_count = shuffled_inputs.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
# They are everything that is remaining.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]


##Save as NPZ files
np.savez('Hockey_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Hockey_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Hockey_data_test', inputs=test_inputs, targets=test_targets)

##Load the npz files
npz = np.load('Hockey_data_train.npz')

train_inputs = npz['inputs'].astype(float)
train_targets = npz['targets'].astype(int)

npz = np.load('Hockey_data_validation.npz')
validation_inputs = npz['inputs'].astype(float)
validation_targets = npz['targets'].astype(int)

npz = np.load('Hockey_data_test.npz')
test_inputs = npz['inputs'].astype(float)
test_targets = npz['targets'].astype(int)


##Define and run the model

input_size = 6
output_size = 22
hidden_layer_size = 500

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 200

max_epochs = 20

early_stopping = tf.keras.callbacks.EarlyStopping(patience=4)

model.fit(train_inputs, train_targets, batch_size = batch_size, epochs = max_epochs, callbacks = [early_stopping],
          validation_data=(validation_inputs, validation_targets), verbose=2)



#Test the model
#test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
#print(test_loss, test_accuracy)


testarr = np.array([[2,3,1,5,0,0]])
results = model.predict(testarr) * 100,2
results = np.array(results)
results

##Export the module
#import pickle

#with open('hockeyModel', 'wb') as file:
    #pickle.dump(model,file)
    
#with open('hockeyScaler', 'wb') as file:
    #pickle.dump(unscaled_inputs_all, file)
    
    
##Export model with TF
model.save("hockeyModel2")

model = tf.keras.models.load_model("hockeyModel2")


##Logistic using sklearn to check for unnecessary features
reg2 = LogisticRegression(multi_class='ovr', solver='liblinear')

reg2.fit(train_inputs, train_targets)

coefDf = pd.DataFrame(reg2.coef_)

tempOdds = []

for i in range(coefDf.shape[1]):
    tempOdds = coefDf.iloc[:,i] ** 2
    coefDf['Odds Ratio ' + str(i)] = tempOdds

new_col_order = [0, 'Odds Ratio 0', 1, 'Odds Ratio 1', 2, 'Odds Ratio 2', 3,'Odds Ratio 3', 4,'Odds Ratio 4', 
                 5,'Odds Ratio 5']

coefDf = coefDf[new_col_order]

coefDf.to_csv("Odds Ratios.csv")