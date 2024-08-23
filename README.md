### Developed by: DODDA JAYASRI
### Register No: 212222240028
### Date:
# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### AIM:
To Implement Linear and Polynomial Trend Estimation Using Python.

### ALGORITHM:
Step 1: Import necessary libraries (NumPy, Matplotlib)

Step 2: Load the dataset

Step 3: Calculate the linear trend values using least square method

Step 4: Calculate the polynomial trend values using least square method

Step 5: End the program
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data= pd.read_csv("Salesforcehistory.csv")
data.head()
data['Date']=pd.to_datetime(data['Date'])
data.info()
X=np.arange(len(data)).reshape(-1,1)
y=data['Close'].values

A - LINEAR TREND ESTIMATION


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
y_linear_predict=regressor.predict(X)

# Graph for Linear trend
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], y, label='Actual Data', color='black')
plt.plot(data['Date'], y_linear_predict, label='Linear Trend', color='blue')
plt.title('Tsla Stock predication')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# print the Linear Trend Equation
print(f"Linear Trend Equation: y = {regressor.coef_[0]:.2f} * x + {regressor.intercept_:.2f}")


B- POLYNOMIAL TREND ESTIMATION


from sklearn.preprocessing import PolynomialFeatures
# polynomial trend for degree 2
degree=2
poly_reg=PolynomialFeatures(degree=degree)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
regressor_poly=LinearRegression()
regressor_poly.fit(X_poly,y)
y_predict_poly_2=regressor_poly.predict(X_poly)

# polynomial trend for degree 3
degree_3=3
poly_reg_3=PolynomialFeatures(degree=degree_3)  # Use degree_3 here
X_poly_3=poly_reg_3.fit_transform(X)
poly_reg_3.fit(X_poly_3,y)
regressor_poly_3=LinearRegression()
regressor_poly_3.fit(X_poly_3,y)
y_predict_poly_3=regressor_poly_3.predict(X_poly_3)

# Graph for polynomial trend
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], y, label='Actual Data', color='black')
plt.plot(data['Date'], y_predict_poly_2, label=f'Polynomial Trend (Degree {degree})', linestyle='-.', color='green')
plt.plot(data['Date'], y_predict_poly_3, label=f'Polynomial Trend (Degree {degree_3})',  color='red')
plt.title('Tsla Stock predication')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# print the Polynomial Equations
print("Polynomial Trend Equation (Degree 2): y = {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    regressor_poly.coef_[2], regressor_poly.coef_[1], regressor_poly.intercept_))
print("Polynomial Trend Equation (Degree 3): y = {:.2f} * x^3 + {:.2f} * x^2 + {:.2f} * x + {:.2f}".format(
    regressor_poly_3.coef_[3], regressor_poly_3.coef_[2], regressor_poly_3.coef_[1], regressor_poly_3.intercept_))
```


### OUTPUT
A - LINEAR TREND ESTIMATION
![Screenshot 2024-08-19 112549](https://github.com/user-attachments/assets/94e2ee96-4e9a-402d-8756-7734cbdfc2c5)

B- POLYNOMIAL TREND ESTIMATION
![Screenshot 2024-08-19 112601](https://github.com/user-attachments/assets/233fdf66-32e6-4e91-a492-6897f96fb7a8)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
