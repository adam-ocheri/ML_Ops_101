import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
Linear Regression is a mathematical function used for describing a linear relationship between a value that depends on other independent variables.
In Machine Learning, it is used to describe the relationship between features and a target - Predictor(s) and a Response variable.
It assumes a linear influence of the features on the target variable - meaning, a "direct", or, continuous effect.

To understand linear regression, imagine a 2D graph with some points scattered on it - and a line that crosses through these points.
In the context of a linear regression setup, that line would describe the best linear path that could traverse through the points, with the closest proximity to 
each point as possible: (Run the file!)
"""

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset to train&test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create LR class instance, train the model, and make prediction
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Log Coefficients
# * The initial starting point of the regression line on the Y axis
print("Y-Intercept: ", regressor.intercept_)
# * The magnitude of change in Y for each 1 unit of X
print("Coefficient: ", regressor.coef_)

#! Visualize the Best Path (training set result)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Best Path')
plt.show()

#! Visualize our model's prediction of the best path (test set results)
plt.scatter(X_test, y_test, color='purple')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Predicted Path')
plt.show()

"""
This is the formula for basic linear regression:

    Y = β0 + β1*X + ε

Each variable here plays a role on influencing the outcome "Y", and it is important to understand the role of each of them:

    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    #Target Variable                #const Y-Intercept              #Slope-Coefficient              #Independent Variable               #Error Term
    *      Y               =               β0              +               β1              *               X                   +               ε
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    # **Y-Intercept**: 
        The value of the `Y-Intercept` represents the initial point that the regression line starts from. Geometrically, it is the point where the 
        regression line intersects the y-axis.

        The `Y-Intercept` is often denoted as `b` or `β0`, and it is a constant term in the linear regression equation. It represents the predicted value of the 
        dependent variable when all independent variables are zero (for example, {X = 0, y = 42}).

    # **Slope Coefficient**:
        The value of the `Slope Coefficient` represents the amount of change in the Target Variable, for every integral change (+-1.0) of the Independent Variables,
        the `X-Features Matrix`. It quantifies the relationship between `y`, the dependent variable, for a unit change in the independent variable, `X`.
        In a simple linear regression model with one independent variable, the slope coefficient is a constant value.

        The value of the slope coefficient is estimated based on the data during the model fitting process. It is not predetermined but is determined by finding the 
        best-fitting line that minimizes the sum of squared differences between the predicted values and the actual values. Each observation in the dataset 
        contributes to the estimation of the slope coefficient, and it is specific to the particular dataset and model.
        (for example,
            x = [0.2, 0.6, 0.9, 1.1]
            y = [2, 3, 5, 7]

            mean_x = sum(x) / len(x)
            mean_y = sum(y) / len(y)

            slope_coefficient = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / sum((xi - mean_x) ** 2 for xi in x)

        )
        In linear regression, the coefficient, often denoted as `m` or `β1`, represents the slope of the line.  The coefficient determines the direction and 
        magnitude of the relationship between the variables. 
        In multiple linear regression, there is a coefficient associated with each independent variable.

    # **Error Term**:
         
"""

"""
* Understanding the Slope Coefficient

Let's say we have a dataset that contains information about houses, including the size of the house in square feet (`X`) and the corresponding 
sale prices in dollars (`y`). We want to build a linear regression model to predict the sale price of a house based on its size.

After fitting the linear regression model to the data, we obtain the following equation:

*    y = 5000 + 200 * X

In this equation, the slope coefficient is 200. It indicates that for every one-unit increase in the size of the house, the predicted sale price will 
increase by $200. So, if we have two houses with a size difference of 100 square feet, we would expect the predicted sale price of the larger house to 
be $20,000 higher than the smaller house.

For example:

*    House 1: Size = 1500 sq. ft.    
*    House 2: Size = 1600 sq. ft.       // $20,000 Difference

Using the linear regression equation, we can calculate the predicted sale prices:

    Predicted price for House 1: 
    *                            `5000 + 200 * 1500 = $305,000`
    Predicted price for House 2: 
    *                            `5000 + 200 * 1600 = $325,000`

The difference in predicted prices is indeed $20,000, reflecting the impact of the slope coefficient.

Remember that the actual values of the slope coefficient and the intercept (5000 in this case) would be determined by the linear regression model based 
on the specific dataset used for training. The example above is just a hypothetical illustration to demonstrate the concept.
"""

"""
* Ordinary Last Squares (OLS)


"""
