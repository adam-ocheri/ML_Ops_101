# Machine Learning 101

This repository documents the learning process of the **Machine Learning A-Z™: AI, Python & R** Udemy course by _Hadelin de Ponteves_

https://www.udemy.com/course/machinelearning/

All files under the `ML_101_Code_N_Datasets` are the original course material, which are freely available at https://www.superdatascience.com/pages/machine-learning

**Table Of Contents**

- [What is Machine Learning](#what-is-machine-learning)

- [The Machine Learning Process](#the-machine-learning-process)

  - [A) Data Pre-processing](#a-data-pre-processing)
    - [1. Import the data](#1-import-the-data)
    - [2. Clean the data](#2-clean-the-data)
    - [3. Splitting the data](#3-splitting-the-data)
  - [B) Modelling](#b-modelling)
    - [1. Build the model](#1-build-the-model)
    - [2. Train the model](#2-train-the-model)
    - [3. Make predictions](#3-make-predictions)
  - [C) Evaluation](#c-evaluation)
    - [1. Calculate performance metrics](#1-calculate-performance-metrics)
    - [2. Make a verdict](#2-make-a-verdict)

- [Basic ML Training Theory](#basic-ml-training-theory)

- [A Word on Data (Analysis)](#a-word-on-data-analysis)
  - [Feature Selection](#1-feature-selection)
  - [Domain Expertise](#2-domain-expertise)
  - [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
  - [Statistical Analysis](#4-statistical-analysis)
  - [Correlation Analysis](#5-correlation-analysis)
  - [Feature Engineering](#6-feature-engineering)
  - [Domain-Specific Metrics](#7-domain-specific-metrics)

## What is Machine Learning

Machine learning is merely the application of advanced statistics operations and formulas on large amounts of data. The result of this operation enables us to make a `prediction`; this prediction can be thought of like a function, which would take similar data as the training data for input, and output the statistically-arrived-at result based on the prediction made, considering all the data the model was trained on.

The various learning models employed by ML Ops are "simply" different tools for different tasks - each using some form of input data and processing to arrive at some desired result, on which we can make a `prediction` - only to repeat the process again, based on the previous results.

This iterative nature of this procedure is effectively the "learning" part within machine learning - where a learning model is being trained and `re-trained` again continuously, constantly re-articulating the results arrived at - on top of the accumulation of all past results.

---

Another way to look at this would be:

Machine learning involves the application of advanced statistical operations and formulas to analyze large amounts of data. By leveraging this process, we can generate predictions that function like a mathematical model. This model takes similar data to the training data as input and produces statistically-derived results based on the learned patterns from the training phase.

The different learning models used in machine learning operations (ML Ops) are diverse tools designed for specific tasks. Each model utilizes distinct techniques for processing input data to achieve desired outcomes, enabling us to make predictions. This iterative process of training and prediction forms the core of machine learning, as models are continuously trained and retrained, accumulating insights from past results.

## The Machine Learning Process

#### A) Data Pre-processing

- ###### 1. Import the data
  - Gathering all of our training and test datasets
- ###### 2. Clean the data
  - Arranging the data properly so that all entries conform to some particular input format
- ###### 3. Splitting the data
  - Out of 100% of our data, 80% of it should go for training, while the remaining 20% are reserved for testing
    - This way, we can later test our model and compare it's results against the actual data defined in the testing-dataset, and evaluate if the model's predictions are sufficiently accurate

#### B) Modelling

- ###### 1. Build the model
  - In this step, we define the architecture of the machine learning model that will be used to make predictions on our data. This involves selecting the appropriate model type, such as decision trees, neural networks, or support vector machines, and configuring its parameters
- ###### 2. Train the model
  - Once the model is built, we need to train it using the training dataset. Training involves feeding the model with input data and the corresponding correct output labels. The model then learns from this data and adjusts its internal parameters to minimize the prediction error
- ###### 3. Make predictions
  - After the model is trained, we can use it to make predictions on new, unseen data. We input the features of the new data into the model, and it generates predictions or classifications based on what it has learned during training.

#### C) Evaluation

- ###### 1. Calculate performance metrics
  - In this step, we assess the performance of the trained model using various evaluation metrics. These metrics can include accuracy, precision, recall, F1 score, and others, depending on the nature of the problem and the type of model
- ###### 2. Make a verdict
  - Based on the performance metrics, we can make a verdict about the effectiveness of the model. This involves analyzing the metrics and determining if the model meets the desired criteria or if further improvements are needed. The verdict helps in deciding whether the model is ready for deployment or if further iterations of the machine learning process are required

## Basic ML Training Theory

Machine learning training involves the use of dependent variables and independent variables. A dependent variable, also known as the target variable or output variable, is the variable we want to predict or understand. It represents the outcome or the value we are interested in. In the context of a car price prediction example, the dependent variable could be the price of a car.

On the other hand, independent variables, also known as features or input variables, are the variables that are used to predict the dependent variable. These variables are considered to have an influence on the dependent variable. In the car price prediction example, the independent variables could include the production year of the car, mileage, engine size, brand, and other relevant features.

The goal of machine learning training is to find a mathematical relationship or pattern between the independent variables and the dependent variable. This relationship is captured by the machine learning model during the training process. The model learns from the input data and makes predictions or estimates about the dependent variable based on the observed patterns in the independent variables.

The training process involves feeding the machine learning algorithm with a labeled dataset, where both the independent variables and the corresponding dependent variable values are provided. The algorithm learns from this data by adjusting its internal parameters to minimize the difference between the predicted values and the actual values of the dependent variable.

Once the training process is completed, the trained model can be used to make predictions on new, unseen data. It takes the values of the independent variables as input and produces predictions for the dependent variable. These predictions can be used for various purposes, such as making informed decisions, understanding patterns and trends, or optimizing processes.

It's important to note that the choice and selection of independent variables play a crucial role in the accuracy and effectiveness of the machine learning model. Domain knowledge and data analysis techniques are often employed to identify the most relevant features that have a significant impact on the dependent variable.

By leveraging machine learning techniques and training models on relevant datasets, we can uncover complex relationships and patterns in the data, enabling us to make accurate predictions and gain valuable insights for various applications.

## A Word on Data (Analysis)

In the field of machine learning, several domain knowledge and data analysis techniques are commonly employed to identify the most relevant features that have a significant impact on the dependent variable. Some of these techniques include:

###### 1. Feature Selection:

Feature selection methods aim to identify the subset of features that are most informative and relevant to the dependent variable. These techniques help in reducing dimensionality and removing irrelevant or redundant features. Examples of feature selection methods include statistical tests, correlation analysis, information gain, and recursive feature elimination.

###### 2. Domain Expertise:

Domain experts possess specific knowledge about the problem domain and can provide insights into which features are likely to be important. Their expertise helps in understanding the underlying relationships between variables and identifying relevant features based on their domain knowledge.

###### 3. Exploratory Data Analysis (EDA):

EDA involves visualizing and exploring the dataset to gain insights into the relationships between variables. Techniques such as scatter plots, histograms, box plots, and correlation matrices can help identify patterns, trends, and potential relationships between variables.

###### 4. Statistical Analysis:

Statistical techniques, such as hypothesis testing, analysis of variance (ANOVA), and chi-square tests, can be used to assess the significance of different variables in relation to the dependent variable. These tests help in identifying features that have a strong association or impact on the outcome of interest.

###### 5. Correlation Analysis:

Correlation analysis measures the strength and direction of the relationship between variables. By calculating correlation coefficients, such as Pearson's correlation coefficient or Spearman's rank correlation coefficient, one can identify variables that are highly correlated with the dependent variable.

###### 6. Feature Engineering:

Feature engineering involves creating new features or transforming existing features to better represent the underlying relationships in the data. This process requires a deep understanding of the problem domain and the specific characteristics of the data. Domain knowledge can guide the creation of relevant features that capture important aspects of the problem.

###### 7. Domain-Specific Metrics:

In some domains, there are specific metrics or measures that are known to be important indicators of the dependent variable. For example, in the healthcare domain, certain clinical measurements or biomarkers may have a direct impact on the prediction of a disease outcome.

It's important to note that the choice of techniques and approaches for feature selection and identification of relevant features depends on the specific problem domain, available data, and the characteristics of the dataset. A combination of domain expertise, data analysis techniques, and iterative experimentation is often used to refine the set of features and improve the performance of machine learning models.

## ML Course Sections Summary

In the world of Machine Learning, there are many fundamental principles to be knowledgeable about across many different fields and domains, from algorithms to statistics and mathematics.

#### Regression

Regression is a supervised learning technique used for predicting continuous numerical values based on input features. It aims to establish a functional relationship between the independent variables (input features) and the dependent variable (output value) by fitting a regression model to the training data.

**Intuitive Description**:

Regression is like drawing a line or curve through scattered points on a graph. It helps us understand the relationship between the input variables and the output variable. With regression, we can make predictions for new input values based on the learned patterns from the training data.

**Technical Explanation**:

In regression, the goal is to find a mathematical function that best represents the relationship between the input features and the output variable. The regression model takes the form:

```python
y = f(x1, x2, ..., xn) + ε
```

where `y` represents the output variable, `x1, x2, ..., xn` represent the input features, `f()` represents the regression function, and `ε` represents the error term. The regression function `f()` can be linear or nonlinear, depending on the complexity of the relationship.

To find the best fit, regression models estimate the coefficients (parameters) of the regression function based on the given training data. The common objective is to minimize the difference between the predicted values and the actual values.

**Foundational Mathematical Formula**:

The most basic form of regression is simple linear regression, which assumes a linear relationship between the input feature x and the output variable y. The formula for simple linear regression is:

```python
y = β0 + β1 * x + ε
```

where `y` is the dependent variable, `x` is the independent variable, `β0` is the y-intercept, `β1` is the slope (coefficient), and `ε` is the error term.

The coefficients β0 and β1 are estimated using various methods such as ordinary least squares (OLS) or gradient descent.

**Code Example**:

Here's a basic code example using scikit-learn's LinearRegression class to perform simple linear regression:

```python
from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Prepare the training data
X_train = [[1], [2], [3], [4], [5]]  # Input feature
y_train = [2, 4, 6, 8, 10]  # Output variable

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions for new input
X_test = [[6], [7], [8]]  # New input feature
y_pred = model.predict(X_test)

print(y_pred)  # Output: [12. 14. 16.]
```

In this example, we create a linear regression model using the `LinearRegression` class. We then provide the training data consisting of input features (`X_train`) and corresponding output values (`y_train`). The model is fitted to the training data using the `fit()` method.

After training, we can use the model to make predictions for new input values (X_test) using the predict() method. The predicted output values (y_pred) are printed, indicating the model's estimation for the given input.

Note that this example demonstrates simple linear regression, but regression techniques can be extended to handle more complex relationships and multiple input features using various algorithms and methods.

###### Related Terms

- **Linear Regression**:
  Linear regression is a statistical modeling technique used to establish a linear relationship between a dependent variable and one or more independent variables. It assumes a linear equation of the form `y = mx + b`, where `y` is the dependent variable, `x` is the independent variable, `m` is the slope, and `b` is the y-intercept. The goal of linear regression is to find the best-fitting line that minimizes the difference between the observed data points and the predicted values.

- **Non-Linear Regression**:
  Non-linear regression is a regression analysis technique used when the relationship between the dependent variable and independent variables cannot be adequately described by a linear equation. In non-linear regression, the relationship is modeled using non-linear functions, such as exponential, logarithmic, polynomial, or trigonometric functions. The parameters of the non-linear function are estimated to fit the data and make predictions.

- **Y-intercept**:
  The y-intercept, often denoted as `b` or `β0`, is a constant term in the linear regression equation that represents the predicted value of the dependent variable when all independent variables are zero. Geometrically, it is the point where the regression line intersects the y-axis. The `y-intercept` provides an initial starting point for the regression line and contributes to the overall linear relationship between the variables.

- **Coefficients / Slope**:
  In the context of regression, coefficients refer to the parameters that determine the relationship between the independent variables and the dependent variable. In linear regression, the coefficient, often denoted as `m` or `β1`, represents the slope of the line. It indicates the change in the dependent variable for a unit change in the independent variable. The coefficient determines the direction and magnitude of the relationship between the variables. In multiple linear regression, there is a coefficient associated with each independent variable.

- **Error Term**:
  The error term, often denoted as ε or residuals, represents the unexplained variation or the discrepancy between the predicted values and the actual values of the dependent variable. It accounts for the factors that are not captured by the regression model. The goal of regression analysis is to minimize the sum of squared errors or residuals, indicating the goodness of fit between the observed data and the regression line.

- **Ordinary Least Squares (OLS)**:
  Ordinary Least Squares is a method used to estimate the coefficients in linear regression models. It aims to find the best-fitting line by minimizing the sum of squared residuals between the observed data and the predicted values. OLS estimates the coefficients that provide the smallest sum of squared residuals, making it a popular and widely used method for linear regression. It provides closed-form solutions for the coefficient estimates, making computations efficient.

- **Gradient Descent**:
  Gradient descent is an optimization algorithm commonly used in machine learning, including regression. It iteratively updates the model parameters to minimize an objective function, often represented by an error or loss function. In regression, gradient descent is used to estimate the optimal values for the coefficients that minimize the sum of squared errors. The algorithm starts with initial estimates for the coefficients and iteratively adjusts them in the direction of steepest descent of the error surface. The process continues until the algorithm converges to the optimal values or reaches a predefined stopping criterion. Gradient descent can handle both linear and non-linear regression problems. It is particularly useful for large datasets or complex models where closed-form solutions like OLS are not feasible.

---

#### Classification

**Intuitive Description**:

Classification is a machine learning task that involves categorizing or assigning predefined labels or classes to input data based on their features. It aims to learn a decision boundary or a classification rule that can accurately predict the class of unseen instances. Classification is used in various applications, such as spam detection, image recognition, sentiment analysis, and disease diagnosis. The goal is to train a model that can generalize well and accurately classify new data points into their respective classes.

**Technical Explanation**:

In classification, the task is to build a model that can learn the relationship between input features and corresponding class labels. The model is trained on a labeled dataset, where each data instance has a set of input features and a known class label. The process involves extracting relevant features, selecting an appropriate algorithm, and training the model using the labeled data. The trained model can then be used to predict the class labels of unseen instances.

The most common classification algorithms include logistic regression, support vector machines (SVM), decision trees, random forests, naive Bayes, and k-nearest neighbors (KNN). These algorithms employ various techniques to learn the decision boundaries, such as optimizing parameters, constructing decision trees, or estimating probabilities. The choice of algorithm depends on the characteristics of the data and the specific requirements of the problem.

**Mathematical Formula and Calculations**:

In binary classification, where there are two classes (e.g., positive and negative), a common approach is logistic regression. It models the relationship between the input features X and the probability of belonging to a particular class Y=1 using the logistic function, also known as the sigmoid function:

```python
p(Y=1 | X) = 1 / (1 + exp(-z))
```

where z is the linear combination of the input features weighted by coefficients:

```javascript
z = b0 + b1*X1 + b2*X2 + ... + bn*Xn
```

1. `p(Y=1 | X)`: This represents the probability of the response variable Y taking the value 1 given the input features X. In logistic regression, we are interested in estimating the probability of a binary outcome (e.g., presence or absence of an event) based on the given features.

2. `z`: The variable z is the linear combination of the input features X, weighted by coefficients. It represents the linear part of the logistic regression equation. Each feature X is multiplied by its corresponding coefficient b and then summed up. The intercept term b0 is added as well. The resulting value z is the input to the sigmoid function.

3. `b0, b1, b2, ..., bn`: These are the coefficients or weights associated with each feature X. They represent the impact or contribution of each feature on the outcome variable Y. The coefficients are estimated during the training process of the logistic regression model.

4. `X1, X2, ..., Xn`: These are the input features or independent variables. Each feature represents a different aspect or attribute of the data that may influence the outcome variable. The logistic regression model learns the relationship between these features and the probability of the outcome.

5. `exp(-z)`: The exp function denotes the exponential function, which raises the mathematical constant e (approximately 2.71828) to the power of -z. This is a crucial step in logistic regression as it transforms the linear combination z into a range between 0 and 1.

6. `1 / (1 + exp(-z))`: This expression represents the sigmoid or logistic function. It takes the transformed z value and maps it to a probability value between 0 and 1. The logistic function ensures that the estimated probability stays within this valid range. When z is large and positive, the probability tends toward 1, and when z is large and negative, the probability tends toward 0.

The logistic regression model learns the optimal values for the coefficients b0, b1, b2, ..., bn during the training process. The objective is to find the coefficients that maximize the likelihood of the observed data. This is typically done using optimization techniques such as maximum likelihood estimation or gradient descent.

During the training process, the logistic regression model adjusts the coefficients to minimize the difference between the predicted probabilities and the actual labels in the training data. The model learns to find the optimal decision boundary that separates the two classes by assigning higher probabilities to instances of one class and lower probabilities to instances of the other class.

Other classification algorithms may have different mathematical formulations and calculations. For example, support vector machines (SVM) aim to find a hyperplane that maximally separates the classes in a high-dimensional feature space, while decision trees use recursive partitioning to create hierarchical decision rules based on feature thresholds.

**Code Example**:

Here's a basic code example using scikit-learn's logistic regression classifier:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
X, y = load_dataset()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression classifier
clf = LogisticRegression()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```

In this example, the dataset is split into training and testing sets using the train_test_split function. Then, a logistic regression classifier is created using LogisticRegression class. The classifier is trained on the training data using the fit method. Predictions are made on the test set using the predict method, and the accuracy of the predictions is evaluated using the accuracy_score function.

---

---

---

---

#### Mean (Average)

#### Fitting

#### Transforming

#### Supervised Learning

...

#### Unsupervised Learning

...

#### Semi-Supervised Learning

...

#### Reinforcement Learning

...

#### Classification

...

#### Regression

...

#### Clustering

...

#### Dimensionality Reduction

...

#### Feature Extraction

...

#### Feature Selection

...

#### Feature Engineering

...

#### Overfitting

...

#### Underfitting

...

#### Bias-Variance Tradeoff

...

#### Cross-Validation

...

#### Validation Set

...

#### Hyperparameters

...

#### Model Evaluation

...

#### Precision

...

#### Recall

...

#### F1 Score

...

#### Accuracy

...

#### Confusion Matrix

...

#### Decision Tree

...

#### Random Forest

...

#### Support Vector Machines (SVM)

...

#### Naive Bayes

...

#### Neural Networks

...

#### Gradient Descent

...

#### Backpropagation

...

#### Regularization

...

#### L1 Regularization (Lasso)

...

#### L2 Regularization (Ridge)

...

#### Ensemble Learning

...

#### Bagging

...

#### Boosting

...

#### AdaBoost

...

#### XGBoost

...

#### K-nearest Neighbors (KNN)

...

#### Principal Component Analysis (PCA)

...

#### K-means Clustering

...

#### Mean Shift Clustering

...

#### Gaussian Mixture Models (GMM)

...

#### Natural Language Processing (NLP)

...

#### Recommender Systems

...

#### Deep Learning

...

#### Convolutional Neural Networks (CNN)

...

#### Recurrent Neural Networks (RNN)

...

#### Generative Adversarial Networks (GANs)

...

"""
Supervised vs. Unsupervised - what's the Difference?

Coefficients

WTF are hyperparameters

TF is the Epsilon-Greedy Strategy
"""
