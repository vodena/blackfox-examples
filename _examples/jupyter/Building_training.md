
# Building
  
### Problem explanation:

Prediction of energy consumption in a building. Try to predict the hourly consumption of electrical energy, hot water, and cold water, based on the date, time of day, outside temperature, outside air humidity, solar radiation, and wind speed.
The data set was created based on problem A of “The Great Energy Predictor Shootout - the first building data analysis and prediction problem” contest, organized in 1993 for the ASHRAE мeeting in Denver, Colorado. The data set itself is located here, in the field building.

This is regression problem and the results  are three outputs, hourly consumption of electrical energy, hot water, and cold water. Model inputs are:
 
* Day,
* Hour,
* Temperature,
* Humidity,
* Solar radiation,
* Wind speed.

### Problem solution:
Data set contains 4208 observations,we have divided the data set in two sets, training set, which contains 3366 observations and test set, which contains 842 observations. We solved problem in two ways, with Python and Black Fox. We measured the model performance with K-cross validation (K=5) and for feature scaling we used min-max scaler. Input day is categorical data, so it was encoded with one hot encoder (to avoid dummy variable trap we ignored, for example monday) and the hour, as cyclic data was encoded with distance of noon. To stop training at the right time we used Early Stopping.


#### Update Keras to latest version:


```python
!pip install keras==2.2.4
```

# Data preprocessing
#### Importing dataset:


```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataframe = pd.read_csv('BuildingData.csv')
```

#### Dataset info:


```python
dataframe.info()
```

#### Dataset description:


```python
dataframe.describe()
```

#### Dataset histogram:


```python
dataframe.hist(figsize=(14,14));
```

#### Corelation heatmap:


```python
sns.heatmap(dataframe.corr(), vmin=0, vmax=1);
```

#### We will separate data frame into matrix X of features and dependent variable which is matrix y.


```python
X = dataframe.iloc[:, 0:14].values
y = dataframe.iloc[:, 14:17].values

# To avoid dummy variable trap we remove column.
X = X[:, 1:]
```

#### Now we are able to split the dataset into the training set and test set.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
```

#### We need to apply feature scaling because we don't wanna have one independent variable dominating another one.


```python
# Min Max Scaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_minMaxScaled = scaler.fit_transform(X_train)
X_test_minMaxScaled = scaler.transform(X_test)
```

# Option 1 - manually finding best ANN:
#### After many times of guessing the parameters for model this are the best one that we have found (you dont see our such enormous effort and huge time to find this parameters).


```python
# Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from keras.callbacks import EarlyStopping

import time
start1 = time.time()

classifier = Sequential()
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 13))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'sigmoid'))
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'auto',
                   #min_delta = 0,
                   patience = 200,
                   verbose = 1,
                   #baseline=0.4,
                   restore_best_weights = True
                  )
classifier.compile(optimizer = 'nadam', loss = 'mean_absolute_error', metrics = ['accuracy'])
classifier.fit(x = X_train_minMaxScaled, y = y_train, validation_split = 0.3, batch_size = 32, epochs = 3000, callbacks = [es], verbose=1)

end1 = time.time()

time1 = int(end1-start1)
minutes1, seconds1= divmod(time1, 60)
hours1, minutes1= divmod(minutes1, 60)
```

#### We just trained our artificial neural network on the training set and now it's time to make the prediction on the test set.


```python
y_pred_trained = classifier.predict(X_test_minMaxScaled)
print("Predicted values are:\n\n", y_pred_trained[:10,:])

t1=y_pred_trained - y_test
t2=np.square(t1)

t21=t2[:,0:1]
t22=t2[:,1:2]
t23=t2[:,2:3]

t31=t21.sum()
t32=t22.sum()
t33=t23.sum()

t41=t31/y_test.shape[0]
t42=t32/y_test.shape[0]
t43=t33/y_test.shape[0]

Rmse1_trained = np.sqrt(t41)
Rmse2_trained = np.sqrt(t42)
Rmse3_trained = np.sqrt(t43)

t_max1 = np.max(y_test[:,0:1])
t_min1 = np.min(y_test[:,0:1])
t_max2 = np.max(y_test[:,1:2])
t_min2 = np.min(y_test[:,1:2])
t_max3 = np.max(y_test[:,2:3])
t_min3 = np.min(y_test[:,2:3])

Prmse1_trained = 100 * (Rmse1_trained / (t_max1 - t_min1))
Prmse2_trained = 100 * (Rmse2_trained / (t_max2 - t_min2))
Prmse3_trained = 100 * (Rmse3_trained / (t_max3 - t_min3))

print("\nTime to manually train one network is ", time1,"seconds(",hours1,"hours,",minutes1,"minutes and ",seconds1,"seconds ).")
print("\nRmse(WBE) = ",Rmse1_trained)
print("Rmse(WBCW) = ",Rmse2_trained)
print("Rmse(WBHW) = ",Rmse3_trained)
print("\nPrmse(WBE) = ",Prmse1_trained)
print("Prmse(WBCW) = ",Prmse2_trained)
print("Prmse(WBHW) = ",Prmse3_trained)
```

# Option 2 - Parameter tuning
#### We have two type of parameters,  the parameters that are leaned from the model during the training and these are the weights, and we have some other parameters that stay fixed, and this parameters are called the hyperparameters. So for example this hyperparameters are the number of epoch, the bach size, the optimizer or the number of neurons in the layers. When we trained our ANN, we trained it with some fixed values of this hyperparameters, but meybe that by taking some other values we would get to better accuracy over all with K cross validation, and so that's what parameter tuning is all about, it consists of finding the best values of this hyperparameters and we are gonna do this with the technique called grid search that will test several combinations of this values and it will return the best choise that leads to the best accuracy with K cross validation.


```python
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

import time
start2 = time.time()

def build_classifier(optimizer):
   classifier = Sequential()
   classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 13))
   classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid'))
   classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'sigmoid'))
   classifier.compile(optimizer = optimizer, loss = 'mean_absolute_error', metrics = ['accuracy'])
   return classifier

Tuning_classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [10, 25, 32],
              'epochs': [100, 500, 3000],
              'optimizer': ['adam', 'rmsprop']
             }

grid_search = GridSearchCV(estimator = Tuning_classifier,
                           param_grid = parameters,
                           #scoring = 'accuracy',
                           cv = 10,
                          )

grid_search = grid_search.fit(X_train_minMaxScaled, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best parameters are :\n", best_parameters)
print("\nBest accuracy is :\n", best_accuracy)


end2 = time.time()

time2 = int(end2-start2)
minutes2, seconds2= divmod(time2, 60)
hours2, minutes2= divmod(minutes2, 60)

print("\nTime training one network is ", time2,"seconds(",hours2,"hours,",minutes2,"minutes and ",seconds2,"seconds).")
```

#### We got our artificial neural network which is in model named "grid_search". Now it's time to make the prediction on the test set.


```python
y_pred_tuning = grid_search.predict_proba(X_test_minMaxScaled)
#print("Predicted values are:\n\n", y_pred_tuning[:10,:])

t1=y_pred_tuning - y_test
t2=np.square(t1)

t21=t2[:,0:1]
t22=t2[:,1:2]
t23=t2[:,2:3]

t31=t21.sum()
t32=t22.sum()
t33=t23.sum()

t41=t31/y_test.shape[0]
t42=t32/y_test.shape[0]
t43=t33/y_test.shape[0]

Rmse1_tuning = np.sqrt(t41)
Rmse2_tuning = np.sqrt(t42)
Rmse3_tuning = np.sqrt(t43)

t_max1 = np.max(y_test[:,0:1]
t_min1 = np.min(y_test[:,0:1]
t_max2 = np.max(y_test[:,1:2]
t_min2 = np.min(y_test[:,1:2]
t_max3 = np.max(y_test[:,2:3]
t_min3 = np.min(y_test[:,2:3]

Prmse1_tuning = 100 * (Rmse1_tuning / (t_max1 - t_min1))
Prmse2_tuning = 100 * (Rmse2_tuning / (t_max2 - t_min2))
Prmse3_tuning = 100 * (Rmse3_tuning / (t_max3 - t_min3))

print("\nTime needed for tuning is ", time2,"seconds(",hours2,"hours,",minutes2,"minutes and ",seconds2,"seconds).")
print("\nRmse(WBE) = ",Rmse1_tuning)
print("Rmse(WBCW) = ",Rmse2_tuning)
print("Rmse(WBHW) = ",Rmse3_tuning)
print("\nPrmse(WBE) = ",Prmse1_tuning)
print("Prmse(WBCW) = ",Prmse2_tuning)
print("Prmse(WBHW) = ",Prmse3_tuning)
```

# Option 3 - Black fox service finding best ANN:

#### Install Black fox service:


```python
!pip install blackfox-1.0.0.0.tar.gz
```

#### Let's run Black Fox service to find best ANN:


```python
# Importing the BF service libraries
from blackfox import BlackFox
from blackfox import KerasOptimizationConfig
from blackfox import OptimizationEngineConfig

blackfox_url = 'http://147.91.204.14:32701'
bf = BlackFox(blackfox_url)

ec = OptimizationEngineConfig(proc_timeout_miliseconds=2000000, population_size=50, max_num_of_generations=10)
c = KerasOptimizationConfig(engine_config=ec, max_epoch = 3000, validation_split=0.3)

import time
start3 = time.time()

# Use CTRL + C to stop optimization
(ann_io, ann_info, ann_metadata) = bf.optimize_keras_sync(
    input_set = X_train,
    output_set = y_train,
    config = c,
    integrate_scaler=False,
    network_path='OptimizedANNBuilding_final.h5'
)

end3 = time.time()
time3 = int(end3-start3)

print('\nann info:')
print(ann_info)

print('\nann metadata:')
print(ann_metadata)
```

#### Data that we transfer to Black Fox service are not scaled, the service will scale the date by its own and when he finish his job he won't change the data, but service ofers us command to scale our data for prediction as he did and we will ofcourse use that.


```python
# Get metadata
meta = bf.get_metadata('OptimizedANNBuilding_final.h5')
scaler_config = meta['scaler_config']

# Scale
x_scaler_config = scaler_config['input']
from sklearn.preprocessing import MinMaxScaler 
min_max_x = MinMaxScaler(feature_range=x_scaler_config['feature_range'])
min_max_x.fit(x_scaler_config['fit'])

X_test_minMaxScaled_withBF = min_max_x.transform(X_test)
#print(X_test_minMaxScaled_withBF[:10,:])
```

#### Prediction:


```python
#Importing ANN model
from keras.models import load_model
model = load_model('OptimizedANNBuilding_final.h5')

#Prediction
y_pred_BF=model.predict(X_test_minMaxScaled_withBF)
#print("Predicted values are:\n\n", y_pred_BF[:10,:])
```

#### Restoring the results on real values:


```python
# Rescale
y_scaler_config = scaler_config['output']
min_max_y = MinMaxScaler(feature_range=y_scaler_config['feature_range'])
min_max_y.fit(y_scaler_config['fit'])

y_pred_BF_realValues = min_max_y.inverse_transform(y_pred_BF)
#print("\nFirst 6 real predicted values are:\n", y_pred_BF_realValues[:6,:])

#y_pred_BF_realValues = mms_y.inverse_transform(y_pred_BF)
#print("\nFirst 6 real predicted values are:\n", y_pred_BF_realValues[:6,:])
```

#### Calculating the error:


```python
t1=np.abs(y_pred_BF_realValues - y_test)
t2=np.square(t1)

t21=t2[:,0:1]
t22=t2[:,1:2]
t23=t2[:,2:3]

t31=t21.sum()
t32=t22.sum()
t33=t23.sum()

t41=t31/y_test.shape[0]
t42=t32/y_test.shape[0]
t43=t33/y_test.shape[0]

Rmse1_BF = np.sqrt(t41)
Rmse2_BF = np.sqrt(t42)
Rmse3_BF = np.sqrt(t43)

t_max1 = np.max(y_test[:,0:1])
t_min1 = np.min(y_test[:,0:1])
t_max2 = np.max(y_test[:,1:2])
t_min2 = np.min(y_test[:,1:2])
t_max3 = np.max(y_test[:,2:3])
t_min3 = np.min(y_test[:,2:3])

Prmse1_BF = 100 * (Rmse1_BF / (t_max1 - t_min1))
Prmse2_BF = 100 * (Rmse2_BF / (t_max2 - t_min2))
Prmse3_BF = 100 * (Rmse3_BF / (t_max3 - t_min3))

minutes3, seconds3= divmod(time3, 60)
hours3, minutes3= divmod(minutes3, 60)
print("\nTime for finding the best ANN by Black Fox service is ", time3,"seconds(",hours3,"hours,",minutes3,"minutes and ",seconds3,"seconds).")

print("\nRoot mean square error (WBE) = ", Rmse1_BF)
print("Root mean square error (WBCW) = ", Rmse2_BF)
print("Root mean square error (WBHW) = ", Rmse3_BF)
print("\nPercentage root mean square error (WBE) = ", Prmse1_BF)
print("Percentage root mean square error (WBCW) = ", Prmse2_BF)
print("Percentage root mean square error (WBHW) = ", Prmse3_BF)
```

# RESULTS AND DISCUSSION


```python
print("\nTime to manually train one network is ", time1,"seconds(",hours1,"hours,",minutes1,"minutes and ",seconds1,"seconds ).")
print("Time needed for tuning is ", time2,"seconds(",hours2,"hours,",minutes2,"minutes and ",seconds2,"seconds).")
print("Time for finding the best ANN by Black Fox service is ", time3,"seconds(",hours3,"hours,",minutes3,"minutes and ",seconds3,"seconds).")
print("\nLet's visualize the results:\n")

objects = ('TrainingANN', 'TuningANN', 'BFservice')
y_pos = np.arange(len(objects))
performance = [time1,time2,time3]
 
plt.bar(y_pos, performance, align='center', alpha=1, color=('blue','red','green'))
plt.xticks(y_pos, objects)
plt.ylabel('Time (seconds)')
plt.title('Time spent on making ANN')
 
plt.show()
```

#### If we want to compare the results by making ANN manually and making it with Black Fox service, we would need to add the time spent in field "TrainingANN" and "TuningANN" in plot above, and that added time would be comparatible with time Black Fox service spent, which are so different, time needed for manually hard work is much larger then time Black Fox spent to make better results, that are given in the plot below.


```python
print("\nRoot mean square errors:\n")
print("Test set RMSE (WBE) on manually train one network is ", Rmse1_trained)
print("Test set RMSE (WBE) on tuning network is  ", Rmse1_tuning)
print("Test set RMSE (WBE) on Black Fox service's optimized network is ", Rmse1_BF)
print("\nTest set RMSE (WBCW) on manually train one network is", Rmse2_trained)
print("Test set RMSE (WBCW) on tuning network is  ", Rmse2_tuning)
print("Test set RMSE (WBCW) on Black Fox service's optimized network is ", Rmse2_BF)
print("\nTest set RMSE (WBHW) on manually train one network is", Rmse3_trained)
print("Test set RMSE (WBHW) on tuning network is  ", Rmse3_tuning)
print("Test set RMSE (WBHW) on Black Fox service's optimized network is ", Rmse3_BF)

n_groups = 3
group_1 = (Rmse1_trained, Rmse1_tuning, Rmse1_BF)
group_2 = (Rmse2_trained, Rmse2_tuning, Rmse2_BF)
group_3 = (Rmse3_trained, Rmse3_tuning, Rmse3_BF)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
print(index)
bar_width = 0.25
opacity = 0.8
 
rects1 = plt.bar(index, group_1, bar_width,
alpha=opacity,
color='b',
label='WBE')
 
rects2 = plt.bar(index + bar_width, group_2, bar_width,
alpha=opacity,
color='g',
label='WBCW')

rects3 = plt.bar(index + bar_width + bar_width, group_3, bar_width,
alpha=opacity,
color='r',
label='WBHW')
 
#plt.xlabel('Person')
plt.ylabel('Error')

plt.title('Root mean square errors')
plt.xticks(index + bar_width, ('TrainingANN', 'TuningANN', 'BFservice'))
plt.legend()
 
plt.tight_layout()
plt.show()
```


```python
print("\Percentage root mean square errors:\n")
print("Test set PRMSE (WBE) on manually train one network is ", Prmse1_trained)
print("Test set PRMSE (WBE) on tuning network is  ", Prmse1_tuning)
print("Test set PRMSE (WBE) on Black Fox service's optimized network is ", Prmse1_BF)
print("\nTest set PRMSE (WBCW) on manually train one network is", Prmse2_trained)
print("Test set PRMSE (WBCW) on tuning network is  ", Prmse2_tuning)
print("Test set PRMSE (WBCW) on Black Fox service's optimized network is ", Prmse2_BF)
print("\nTest set PRMSE (WBHW) on manually train one network is", Prmse3_trained)
print("Test set PRMSE (WBHW) on tuning network is  ", Prmse3_tuning)
print("Test set PRMSE (WBHW) on Black Fox service's optimized network is ", Prmse3_BF)

n_groups = 3
group_1 = (Prmse1_trained, Prmse1_tuning, Prmse1_BF)
group_2 = (Prmse2_trained, Prmse2_tuning, Prmse2_BF)
group_3 = (Prmse3_trained, Prmse3_tuning, Prmse3_BF)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
 
rects1 = plt.bar(index, group_1, bar_width,
alpha=opacity,
color='b',
label='WBE')
 
rects2 = plt.bar(index + bar_width, group_2, bar_width,
alpha=opacity,
color='g',
label='WBCW')

rects3 = plt.bar(index + bar_width + bar_width, group_3, bar_width,
alpha=opacity,
color='r',
label='WBHW')
 
#plt.xlabel('Person')
plt.ylabel('Error ( % )')

plt.title('Percentage root mean square errors')
plt.xticks(index + bar_width, ('TrainingANN', 'TuningANN', 'BFservice'))
plt.legend()
 
plt.tight_layout()
plt.show()
```

#### Although we measured this three options, actually they are not so comparable, because in Python we had a man sitting in office and programming those neural networks(option 1 and 2) while in Black Fox service (option 3), he imported the same data set and the service did the rest, while he went to rest or dring coffe, for example, so actually, in Black Fox service he wrote few lines of code and thats all of hard work. Results in the given plots above speak for themself. As you can see, Black Fox service gave better results in less time and effort to create approximate results in Python as Black Fox did is immeasurably large.
