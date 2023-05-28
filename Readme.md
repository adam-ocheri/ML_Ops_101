# Machine Learning 101

This repository documents the learning process of the **Machine Learning A-Z™: AI, Python & R** Udemy course by _Hadelin de Ponteves_

https://www.udemy.com/course/machinelearning/

All files under the `ML_101_Code_N_Datasets` are the original course material, which are freely available at https://www.superdatascience.com/pages/machine-learning

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

1. **Feature Selection**: Feature selection methods aim to identify the subset of features that are most informative and relevant to the dependent variable. These techniques help in reducing dimensionality and removing irrelevant or redundant features. Examples of feature selection methods include statistical tests, correlation analysis, information gain, and recursive feature elimination.

2. **Domain Expertise**: Domain experts possess specific knowledge about the problem domain and can provide insights into which features are likely to be important. Their expertise helps in understanding the underlying relationships between variables and identifying relevant features based on their domain knowledge.

3. **Exploratory Data Analysis (EDA)**: EDA involves visualizing and exploring the dataset to gain insights into the relationships between variables. Techniques such as scatter plots, histograms, box plots, and correlation matrices can help identify patterns, trends, and potential relationships between variables.

4. **Statistical Analysis**: Statistical techniques, such as hypothesis testing, analysis of variance (ANOVA), and chi-square tests, can be used to assess the significance of different variables in relation to the dependent variable. These tests help in identifying features that have a strong association or impact on the outcome of interest.

5. **Correlation Analysis**: Correlation analysis measures the strength and direction of the relationship between variables. By calculating correlation coefficients, such as Pearson's correlation coefficient or Spearman's rank correlation coefficient, one can identify variables that are highly correlated with the dependent variable.

6. **Feature Engineering**: Feature engineering involves creating new features or transforming existing features to better represent the underlying relationships in the data. This process requires a deep understanding of the problem domain and the specific characteristics of the data. Domain knowledge can guide the creation of relevant features that capture important aspects of the problem.

7. **Domain-Specific Metrics**: In some domains, there are specific metrics or measures that are known to be important indicators of the dependent variable. For example, in the healthcare domain, certain clinical measurements or biomarkers may have a direct impact on the prediction of a disease outcome.

It's important to note that the choice of techniques and approaches for feature selection and identification of relevant features depends on the specific problem domain, available data, and the characteristics of the dataset. A combination of domain expertise, data analysis techniques, and iterative experimentation is often used to refine the set of features and improve the performance of machine learning models.
