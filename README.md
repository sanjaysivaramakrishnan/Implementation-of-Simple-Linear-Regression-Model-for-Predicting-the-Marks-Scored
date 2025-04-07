# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sanjay sivaramakrishnan M
RegisterNumber:  212223240151
*/

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error   
from sklearn.model_selection import train_test_split   
from sklearn.linear_model import LinearRegression
data = pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_machine_learning\data_sets\student_scores.csv')
data.head()
data.info()
data.isnull().sum()
plt.scatter(data['Hours'],data['Scores'])
plt.xlabel('Hours')
_ = plt.ylabel('Scores')
X  = data.iloc[:,:-1]
y  = data.iloc[:,-1]
X
y
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
model = LinearRegression()
model
model.fit(X_train,y_train)
# x_values = np.linspace(start=2, stop=8, num=1000)
y_predict = model.predict(X_test)     
print('Model coefficient (m)',model.coef_)
print('Model intercept (b)',model.intercept_)
mse = mean_squared_error(y_test,y_predict)
print('Mean square error = ',mse)
mae = mean_absolute_error(y_test,y_predict)
print('mean absolute error = ',mae)
rmse = np.sqrt(mae)
print('RMSE = ',rmse)
plt.scatter(X_test,y_test,color='#fc0356')
plt.plot(X_test,y_predict,color='green')
# plt.legend()
plt.title('Test set(H vs S)')
plt.xlabel('Hours')
_=plt.ylabel('Scores')
x = np.array([[13]])
model.predict(x)


```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
