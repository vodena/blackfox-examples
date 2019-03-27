
# Churn
  
### Problem explanation:

The customer churn, also known as customer attrition, refers to the phenomenon whereby a customer leaves a company. Some studies confirmed that acquiring new customers can cost five times more than satisfying and retaining existing customers. As a matter of fact, there are a lot of benefits that encourage the tracking of the customer churn rate, for example, marketing costs to acquire new customers are high. Therefore, it is important to retain customers so that the initial investment is not wasted, It has a direct impact on the ability to expand the company, etc.

A bank is investigating a very high rate of customer leaving the bank. Here is a 10.000 records data set to investigate and predict which of the customers are more likely to leave the bank soon. The data set itself is located here, in the field Artificial_Neural_Networks.

This is classification problem and the results are two outputs, customer will leave the bank or he will not. Model inputs are:
 
* Credit score,
* Geography,
* Gender,
* Age,
* Tenure,
* Balance,
* Number of products,
* Has credit card,
* Is active member,
* Estimated salary.

### Problem solution:
Data set contains 10000 observations,we have divided the data set in two sets, training set, which contains 8000 observations and test set, which contains 2000 observations. We solved problem in two ways, with Python and Black Fox. Model performance was measured with K-cross validation (K=5) and for feature scaling we used min-max scaler. Inputs geography and gender are categorical data, so they were encoded with one hot encoder (to avoid dummy variable trap, for geography we ignored, for example Germany and for gender we ignored female). To stop training at the right time we used Early Stopping.

#### Update Keras to latest version:


```python
!pip install keras==2.2.4
```

# Data preprocessing
#### Importing data frame:


```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the date as data frame wich we will import with pandas using the read_csv function.
dataframe = pd.read_csv('Churn_Modelling.csv')
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
dataframe.hist(figsize=(10,10));
```

#### Corelation heatmap:


```python
sns.heatmap(dataframe.corr(), vmin=0, vmax=1);
```

####  We will separate data frame into matrix X of features(where we will trow first 3 columns because they are useless) and dependent variable which is matrix y.  


```python
X = dataframe.iloc[:, 3:13].values
y = dataframe.iloc[:, 13:14].values
```

#### The next code section is about splitting the data set into the training set and test set but before we do that we must pay attention on categorical variables in our matrix of features and there for we need to encode them.


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()

# Encoding geography( countries )
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()

# Encoding gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# OneHotEncoding the countries to make dummy variables for this categorical variable.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# To avoid dummy variable trap we remove for example countre Germany.
X = X[:, 1:]
```

#### Now we are able to split the dataset into the training set and test set.



```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

# Black fox service finding best ANN

#### Install Black fox service:


```python
!pip install blackfox-1.0.0.tar.gz
```

#### Let's run Black Fox service to find best ANN:


```python
# Importing the BF service libraries and other libraries
from blackfox import BlackFox
from blackfox import KerasOptimizationConfig
from blackfox import OptimizationEngineConfig
import h5py
#from keras.models import load_model
#import numpy as np
#import pandas as pd

blackfox_url = 'http://147.91.204.14:32701'
bf = BlackFox(blackfox_url)

ec = OptimizationEngineConfig(proc_timeout_miliseconds=2000000, population_size=50, max_num_of_generations=20)
c = KerasOptimizationConfig(engine_config=ec, max_epoch = 3000, validation_split=0.3)

# Use CTRL + C to stop optimization
(ann_io, ann_info, ann_metadata) = bf.optimize_keras_sync(
    input_set = X_train,
    output_set = y_train,
    config = c,
    integrate_scaler=False,
    network_path='OptimizedChurnWithBF.h5'
)

print('\nann info:')
print(ann_info)

print('\nann metadata:')
print(ann_metadata)
```

#### Data that we transfer to Black Fox service are not scaled, the service will scale the date by its own and when he finish his job he won't change the data, but service ofers us command to scale our data for prediction as he did and we will ofcourse use that.


```python
# Get metadata
meta = bf.get_metadata('OptimizedChurnWithBF.h5')
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
# Importing ANN model
from keras.models import load_model
model = load_model('OptimizedChurnWithBF.h5')

# Predicted values
y_pred_BF=model.predict(X_test_minMaxScaled_withBF)
#print("Predicted values are:\n\n", y_pred_BF[:20,:])
```

#### Restoring the results on real values:


```python
# Rescale
y_scaler_config = scaler_config['output']
min_max_y = MinMaxScaler(feature_range=y_scaler_config['feature_range'])
min_max_y.fit(y_scaler_config['fit'])

y_pred_BF_realValues = min_max_y.inverse_transform(y_pred_BF)
#print("\nFirst 6 real predicted values are:\n", y_pred_BF_realValues[:6,:])
```

#### Calculating the error:


```python
y_pred_BF_realValues_rounded = (y_pred_BF_realValues[:,0] > y_pred_BF_realValues[:,1])
y_pred_BF_realValues_rounded = np.where(y_pred_BF_realValues_rounded == True, 1, 0)
y_test_for_confusionMatrix = (y_test[:,0] > y_test[:,1])
y_test_for_confusionMatrix = np.where(y_test_for_confusionMatrix == True, 1, 0)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_for_confusionMatrix, y_pred_BF_realValues_rounded)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_winner)

errorOnTestSetBF = 100*(cm[0,1]+cm[1,0])/y_test.shape[0]

print("\nWe got confusion matrix:\n",cm)
print("\nTest set error for finding the best ANN by Black Fox service, which we can read in confusion matrix is",errorOnTestSetBF,"%.")
```
