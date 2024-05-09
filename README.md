# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Thrinesh Royal.T
RegisterNumber:  212223230226
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(Y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
*/
```

## Output:
### Dataset:
![image](https://github.com/Jeshwanthkumarpayyavula/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742402/83379170-1687-4ab3-a6d2-2ce8348e4105)
### Categories:
![image](https://github.com/Jeshwanthkumarpayyavula/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742402/05990e85-c6c2-4a0d-9d2c-f435d38cc6a7)
![image](https://github.com/Jeshwanthkumarpayyavula/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742402/f98be794-7139-46b1-a46a-cd6322599348)
### X&Y values:
![image](https://github.com/Jeshwanthkumarpayyavula/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742402/4431530a-ca5d-4300-85b2-ae5e974908e7)
### Accurcay and y_pred:
![image](https://github.com/Jeshwanthkumarpayyavula/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742402/d796f427-293a-451c-9f8e-f99e6cb386e8)






## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

