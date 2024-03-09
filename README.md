# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 2.Set variables for assigning dataset values. 3.Import linear regression from sklearn. 4.Assign the points for representing in the graph. 5.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sanjay sivaramakrishnan M
RegisterNumber:  212223240151
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/student_scores.csv")
df.head()
```
```
df.tail()
```
```
#segregating data to variables
X=df.iloc[:,:-1].values
X
```
```
Y=df.iloc[:,1].values
Y
```
```
#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
```
```
#displaying predicted values
Y_pred
```
```
Y_test
```
```
#graph plot for training data
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours VS Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title("Hours VS Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse=mean_squared_error(Y_test,Y_pred)
print('MSE =',mse)
```
```
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE =',mae)
```
```
rmse=np.sqrt(mse)
print('RMSE =',rmse)
```

## Output:
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/20a1eced-0ae2-4e69-8147-3bbce974770a)
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/dc4bd297-e9e4-4325-8fcd-5534088a1c9d)
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/11734e09-debc-44fe-9c71-0f6ff4b9d5c1)
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/80a9d76c-03a8-4f6e-9db9-79f06bfe5a5d)
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/9fa5fd7d-ee12-40d0-b800-2501073e139b)
![image](https://github.com/SanjayBalaji0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145533553/727ef88b-3667-49c3-979d-3a7f64f48128)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
