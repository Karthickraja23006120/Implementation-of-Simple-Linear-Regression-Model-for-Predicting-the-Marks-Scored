# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Karthick Raja K
RegisterNumber:  212223240066
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("student_scores.csv")
df

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

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

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![Screenshot 2024-08-29 113315](https://github.com/user-attachments/assets/e622a890-f91e-4389-991b-8fda4c7410ae)
![Screenshot 2024-08-29 113327](https://github.com/user-attachments/assets/437e83cb-f4b4-4338-8e78-1430472f115e)
![Screenshot 2024-08-29 113337](https://github.com/user-attachments/assets/62759c45-caf1-4548-8259-5b9a54434319)
![Screenshot 2024-08-29 113348](https://github.com/user-attachments/assets/56f3fa9c-7a87-4a77-85e1-0b8c8fa595a3)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
