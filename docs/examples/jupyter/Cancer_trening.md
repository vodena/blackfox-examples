
# Cancer
  
### Problem explanation:

Like other cancers, breast cancer is an uncontrolled growth of breast cells. Breast cancer occurs as a result of abnormal changes or mutation in the genes responsible for regulating the healthy growth of breast. The genes are in each cell’s nucleus, which acts as the “controller” of concern cell.

This abnormal tumour-like growth can be benign (not cancerous) or malignant (cancerous property). Benign tumours are close to normal in appearance, they grow comparatively slowly, and they do not invade or spread to nearby tissue and other parts of the body. As malignant cells have the potential to grow as cancer If they are left unchecked or untreated, they eventually can spread to nearby tissue and beyond an original tumour to other parts of the body.
 We need to classify a tumor as either benign or malignant based on cell descriptions gathered by microscopic examination. 
The data was originally obtained from the University of Wisconsin Hospitals, Madison, from Dr. William H. Wolberg. Data set has 699 observations, all inputs are continuous, 65.5% of the examples are benign. The dataset itself is located here, in the field cancer.

This is classification problem and the results are two outputs, benign or malign. Model inputs are:
 
* Clump thickness,
* Uniformity of cell size,
* Uniformiti of cell shape,
* Marginal adhesion,
* Single epithelial cell size,
* Bare nuclei,
* Bland chromatin,
* Normal nucleoli,
* Mitoses.

### Problem solution:
Data set contains only 699 observations, so it is relatively small data set. We have divided the data set in two sets, training set, which contains 599 observations and test set, which contains 100 observations. We solved problem in two ways, with Python and Black Fox. We measured the model performance with K-cross validation (K=5) and for feature scaling we used min-max scaler. To stop training at the right time we used Early Stopping.
 

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

# Importing the date as data frame wich we will import with pandas using the read_csv function.
dataframe = pd.read_csv('CancerData.csv')
```

#### Dataset info:


```python
dataframe.info()
```

#### Dataset description:


```python
dataframe.describe()
```

#### Dataset histogram :


```python
dataframe.hist(figsize=(10,10));
```

#### Corelation heatmap:


```python
sns.heatmap(dataframe.corr(), vmin=0, vmax=1);
```

####  We will separate data frame into matrix X of features and dependent variable which is matrix y.  


```python
X = dataframe.iloc[:, 0:9].values
y = dataframe.iloc[:, 9:11].values
```

#### We dont need to scale our data because they are already scaled, so we can split the dataset into the training set and test set.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 1)
```

# Option 1 - manually finding best ANN:
#### After many times of guessing the parameters for model this are the best one that we have found (you dont see our such enormous effort and huge time to find this parameters).


```python
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from keras.callbacks import EarlyStopping

import time
start1 = time.time()

classifier = Sequential()
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 9))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'sigmoid'))
es = EarlyStopping(monitor = 'val_loss',
                   mode = 'auto',
                   #min_delta = 1e-2,
                   patience = 150,
                   verbose = 1,
                   #baseline=0.4,
                   restore_best_weights = True
                  )
classifier.compile(optimizer = 'rmsprop', loss = 'mean_absolute_error', metrics = ['accuracy'])
classifier.fit(x = X_train, y = y_train, validation_split = 0.3, batch_size = 32, epochs = 3000, callbacks = [es], verbose=1)

end1 = time.time()

time1 = int(end1-start1)
minutes1, seconds1= divmod(time1, 60)
hours1, minutes1= divmod(minutes1, 60)
```

#### We just trained our artificial neural network on the training set and now it's time to make the prediction on the test set.


```python
y_pred_trained = classifier.predict(X_test)
#print("Predicted values are:\n\n", y_pred_trained[:10,:])

y_winner = (y_pred_trained[:,0] > y_pred_trained[:,1])
y_winner = np.where(y_winner == True, 1, 0)
y_winner_test = (y_test[:,0] > y_test[:,1])
y_winner_test = np.where(y_winner_test == True, 1, 0)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_winner, y_winner_test)

errorOnTestSetTrained = 100*(cm[0,1]+cm[1,0])/y_test.shape[0]

print("\nTime to manually train one network is ", time1,"seconds(",hours1,"hours,",minutes1,"minutes and ",seconds1,"seconds ).")
print("\nWe got confusion matrix:\n",cm)
print("\nTest set error on manually train one network, which we can read in confusion matrix is",errorOnTestSetTrained,"%.")
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
    classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 9))
    classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'mean_absolute_error', metrics = ['accuracy'])
    return classifier

Tuning_classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [10, 25, 32],
              'epochs' : [100, 500, 3000],
              'optimizer' : ['adam','rmsprop']}

grid_search = GridSearchCV(estimator=Tuning_classifier,
                           param_grid=parameters,
                           #scoring='accurasy',
                           cv=10
                          )

grid_search = grid_search.fit(X_train, y_train)

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
y_pred_tuning = grid_search.predict_proba(X_test)
#print("Predicted values are:\n\n", y_pred_tuning[:10,:])

y_pred_rounded_tuning = (y_pred_tuning[:,0] > y_pred_tuning[:,1])
y_pred_rounded_tuning = np.where(y_pred_rounded_tuning == True, 1, 0)
y_winner_test_tuning = (y_test[:,0] > y_test[:,1])
y_winner_test_tuning = np.where(y_winner_test_tuning == True, 1, 0)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_winner_test_tuning, y_pred_rounded_tuning)

errorOnTestSetTuning = 100*(cm[0,1]+cm[1,0])/y_test.shape[0]

print("\nTime needed for tuning is ", time2,"seconds(",hours2,"hours,",minutes2,"minutes and ",seconds2,"seconds).")
print("\nWe got confusion matrix:\n",cm)
print("\nTest set error with tuning, which we can read in confusion matrix is",errorOnTestSetTuning,"%.")
```

# Option 3 - Black fox service finding best ANN:

#### Install Black fox service:


```python
!pip install blackfox-1.0.0.tar.gz
```

#### Let's run Black Fox service to find best ANN:


```python
# Importing the BF service libraries
from blackfox import BlackFox
from blackfox import KerasOptimizationConfig
from blackfox import OptimizationEngineConfig
import h5py
#from keras.models import load_model
#import numpy as np
#import pandas as pd

blackfox_url = 'http://147.91.204.14:32701'
bf = BlackFox(blackfox_url)

ec = OptimizationEngineConfig(proc_timeout_miliseconds=2000000, population_size=50, max_num_of_generations=10)
c = KerasOptimizationConfig(engine_config=ec, max_epoch = 3000, validation_split=0.1)

import time
start3 = time.time()

# Use CTRL + C to stop optimization
(ann_io, ann_info, ann_metadata) = bf.optimize_keras_sync(
    input_set = X_train,
    output_set = y_train,
    config = c,
    integrate_scaler=False,
    network_path='OptimizedANNCancer_final.h5'
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
meta = bf.get_metadata('OptimizedANNCancer_final.h5')
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
model = load_model('OptimizedANNCancer_final.h5')

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
y_winner_BF = (y_pred_BF_realValues[:,0]>y_pred_BF[:,1])
y_winner_BF = np.where(y_winner_BF == True, 1, 0)
y_winner_test = (y_test[:,0]>y_test[:,1])
y_winner_test = np.where(y_winner_test == True, 1, 0)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_winner_BF, y_winner_test)

errorOnTestSetBF = 100*(cm[0,1]+cm[1,0])/y_test.shape[0]

minutes3, seconds3= divmod(time3, 60)
hours3, minutes3= divmod(minutes3, 60)

print("\nTime for finding the best ANN by Black Fox service is ", time3,"seconds(",hours3,"hours,",minutes3,"minutes and ",seconds3,"seconds).")
print("\nWe got confusion matrix:\n",cm)
print("\nTest set error for finding the best ANN by Black Fox service, which we can read in confusion matrix is",errorOnTestSetBF,"%.")
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
print("\nTest set error on manually train one network, which we can read in confusion matrix is",errorOnTestSetTrained,"%.")
print("Test set error with tuning, which we can read in confusion matrix is",errorOnTestSetTuning,"%.")
print("Test set error for finding the best ANN by Black Fox service, which we can read in confusion matrix is",errorOnTestSetBF,"%.")
print("\nLet's visualize the results:\n")

objects = ('TrainingANN', 'TuningANN', 'BFservice')
y_pos = np.arange(len(objects))
performance = [errorOnTestSetTrained,errorOnTestSetTuning,errorOnTestSetBF]
 
plt.bar(y_pos, performance, align='center', alpha=1, color=('blue','red','green'))
plt.xticks(y_pos, objects)
plt.ylabel('Error (%)')
plt.title('Test set error')
 
plt.show()
```

#### Although we measured this three options, actually they are not so comparable, because in Python we had a man sitting in office and programming those neural networks(option 1 and 2) while in Black Fox service (option 3), he imported the same data set and the service did the rest, while he went to rest or dring coffe, for example, so actually, in Black Fox service he wrote few lines of code and thats all of hard work. Results in the given plots above speak for themself. As you can see, Black Fox service gave better results in less time and effort to create approximate results in Python as Black Fox did is immeasurably large.
