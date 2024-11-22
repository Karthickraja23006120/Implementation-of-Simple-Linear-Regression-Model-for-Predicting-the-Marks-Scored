# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
### Karthick Raja K(212223240066)
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## OUTPUT:

![ml4](https://user-images.githubusercontent.com/94222288/200177597-e6ff825e-710a-40ec-842d-50233234b4d3.png)

![mlh](https://user-images.githubusercontent.com/94222288/203804680-9b787e90-79ac-4ddf-a9d8-ec03b8b88ad2.png)

![mlt](https://user-images.githubusercontent.com/94222288/203804727-227f8f8c-d13f-4904-9a3e-df48a2f8e84f.png)


![ml5](https://user-images.githubusercontent.com/94222288/200177609-a5c4987a-11fa-4426-92a8-aced68c0eb61.png)

![ml7](https://user-images.githubusercontent.com/94222288/200177616-98277779-5896-480e-b9ca-702efb43b4de.png)


![ml8](https://user-images.githubusercontent.com/94222288/200177622-f15e4f5e-0163-47f1-80b2-936d0fd1d347.png)

![ml9](https://user-images.githubusercontent.com/94222288/200177626-8323e106-b6de-4688-8186-47e015923feb.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
