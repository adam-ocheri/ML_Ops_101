# Data Preprocessing

- [Data Preprocessing Overview](#data-preprocessing)

  - [Data Cleaning](#1-data-cleaning)
  - [Data Integration](#2-data-integration)
  - [Data Transformation](#3-data-transformation)
  - [Feature Selection and Dimensionality Reduction](#4-feature-selection-and-dimensionality-reduction)
  - [Data Discretization](#5-data-discretization)
  - [Data Splitting](#6-data-splitting)
  - [Data Normalization](#7-data-normalization)

- [Feature Scaling](#feature-scaling)
  - [Standardization](#1-standardization-z-score-normalization)
  - [Normalization](#2-normalization-min-max-scaling)
- [Basic Data Preprocessing](#basic-data-preprocessing)
  - [1. Importing the libraries](#1-importing-the-libraries)
  - [2. Importing Dataset & Segregating the dependent variable from the independent variables](#2-importing-the-dataset---and-segregating-the-dependent-variable-from-the-independent-variables)
  - [3. Split the dataset into training and testing sets](#3-split-the-dataset-into-training-and-testing-sets)

Data preprocessing is an essential step in the data analysis and machine learning pipeline. It refers to the process of preparing and transforming raw data into a clean, structured, and suitable format for further analysis or machine learning algorithms. The quality of the preprocessing greatly affects the accuracy and effectiveness of the subsequent analysis or modeling tasks.

Data preprocessing typically involves several steps, which may vary depending on the specific dataset and the goals of the analysis. Here are some common data preprocessing techniques:

#### 1. Data Cleaning:

This step involves handling missing data, dealing with outliers, and correcting any inconsistencies or errors in the dataset. Missing data can be imputed or removed, outliers can be detected and handled appropriately, and errors can be corrected through techniques such as data validation or data profiling.

#### 2. Data Integration:

In many cases, data comes from multiple sources or in different formats. Data integration involves combining data from various sources and resolving any inconsistencies or conflicts in data structures, variable naming conventions, or data types.

#### 3. Data Transformation:

Data transformation aims to convert the data into a suitable format for analysis. This may involve scaling or normalizing numerical variables, encoding categorical variables, handling date and time formats, or transforming skewed distributions through techniques like logarithmic or power transformations.

#### 4. Feature Selection and Dimensionality Reduction:

In datasets with a large number of features, feature selection techniques can be applied to identify the most relevant and informative features. Dimensionality reduction methods, such as principal component analysis (PCA) or t-distributed stochastic neighbor embedding (t-SNE), can be used to reduce the number of variables while preserving important information.

#### 5. Data Discretization:

Data discretization involves converting continuous variables into categorical or ordinal variables by dividing them into bins or intervals. This can be useful for certain types of analyses or algorithms that require categorical or discrete input.

#### 6. Data Splitting:

Before applying machine learning algorithms, the dataset is often split into training and testing subsets. This allows the model to be trained on a portion of the data and evaluated on unseen data to assess its performance.

#### 7. Data Normalization:

As mentioned earlier, feature scaling techniques like standardization or normalization are applied to bring the features to a consistent scale, reducing the impact of varying scales on the analysis or modeling process.

By performing these preprocessing steps, the data is transformed into a clean, consistent, and well-prepared format, ready for analysis, visualization, or machine learning tasks. Data preprocessing helps improve the quality of the results, reduces bias, and enhances the overall reliability of data-driven analyses and models.

## Feature Scaling

Feature scaling is a data preprocessing technique used in machine learning to standardize or normalize the numerical features of a dataset. It involves transforming the values of the features to a specific range or distribution. The goal of feature scaling is to ensure that all features have similar scales, which can be beneficial for certain machine learning algorithms.

When working with features that have different scales, some algorithms may give more weight or importance to features with larger values, causing the model to be biased towards those features. By scaling the features, you can mitigate the impact of different scales and ensure that each feature contributes more equally to the learning process.

There are two common methods of feature scaling:

###### 1. Standardization (Z-score normalization):

This method transforms the features so that they have a mean of zero and a standard deviation of one. It subtracts the mean value of the feature from each data point and divides by the standard deviation. The formula for standardization is:

```c
X_scaled = (X - mean(X)) / std(X)
```

Standardization makes the data centered around zero, with a standard deviation of one.

###### 2. Normalization (Min-Max scaling):

This method scales the features to a specific range, typically between 0 and 1. It subtracts the minimum value of the feature from each data point and divides by the range (maximum value minus minimum value). The formula for normalization is:

```c
X_scaled = (X - min(X)) / (max(X) - min(X))
```

Normalization maps the data to a fixed range, preserving the relative relationships between the values.

Both standardization and normalization are effective techniques for feature scaling, and the choice between them depends on the specific requirements of your problem and the characteristics of your data.

By applying feature scaling, you can improve the performance and convergence of certain machine learning algorithms, such as those based on gradient descent optimization, and ensure that the features contribute more fairly to the model's training process.

## Basic Data Preprocessing

In general, data preprocessing can be divided into 3 major steps:

1. Importing The Libraries
2. Importing the dataset
3. Splitting the dataset to a train set and test set

This may often be an over-simplification of this procedure, but any basic or complex ML setup would consist of this basic setup at it's core.
This code snippet provides a basic data preprocessing template that can be used and expanded upon as needed.
However, depending on the specific requirements of the dataset, additional steps such as data cleaning, handling missing data, feature scaling, etc., may be required.

_(Code is also available in `1-DataProcessingTemplate.py`)_

```python
# 1. import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# 2. segregate the dependant variable from the independent variables
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 3. Split the dataset to Training and Testing sets
X_train, X_test, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=0)
```

Let us explore this code, step by step.

#### 1. Importing the libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
```

These libraries cover most of our data processing/preprocessing manipulations and plots and make up most of our basic ML tech stack.

###### Libraries Overview

Here is a brief explanation of the purpose of each one of these libraries:

- **NumPy - Matrices & Math:** NumPy is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. NumPy is widely used in data analysis, numerical computations, and machine learning due to its fast array processing capabilities.

- **Matplotlib - Visualize Data:** Matplotlib is a plotting library for creating static, animated, and interactive visualizations in Python. It provides a wide range of customizable plots, including line plots, scatter plots, bar plots, histograms, and more. Matplotlib is commonly used for data visualization tasks, enabling users to create clear and informative plots to explore and present their data.

- **Pandas - Manipulate Data:** Pandas is a powerful data manipulation and analysis library for Python. It provides easy-to-use data structures, such as DataFrames, for efficiently handling structured data. With Pandas, you can load, manipulate, filter, and transform data, perform operations like merging and grouping, handle missing data, and more. It is widely used for data preprocessing, data cleaning, exploratory data analysis, and data wrangling tasks.

- **Scikit-Learn - Data & ML Ops:** Scikit-learn, often abbreviated as sklearn, is a popular machine learning library in Python. It provides a wide range of machine learning algorithms and tools for tasks such as classification, regression, clustering, dimensionality reduction, model selection, and evaluation. Scikit-learn is known for its user-friendly interface and extensive documentation, making it accessible to both beginners and experienced practitioners. It also integrates well with other scientific libraries, such as NumPy and Pandas, for seamless data processing and model building pipelines.

#### 2. Importing the Dataset - and Segregating the dependent variable from the independent variables

```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

Here we first import the raw dataset, and then we are using the **DataFrame** object from `pandas` to extract different ranges from within this 2 Dimensional dataset.

###### Understanding iloc and DataFrame Indexing

The `iloc` method from the pandas DataFrame object is an extremely flexible function that can take index members or even index ranges as arguments to easily filter a list of entries in 2-Dimensional tabular data. This method allows us to select rows and columns in a DataFrame using integer-based indexing.

The arguments passed to this function can be index numbers or Python ranges. A range is denoted by a colon `:`, which by itself describes the entire range.

- The number on the left of the colon represents the `Lower Bound`, the starting point of the range (inclusive).
- The number on the right of the colon represents the `Upper Bound`, the ending point of the range (exclusive).

For example, this code:

```python
my_list = [32, 64, 128, 256, 512, 1024]
...
my_list[2:5]
```

will extract elements from `my_list` starting at index 2 (inclusive) and ending at index 5 (exclusive). The resulting range will include elements at indices 2, 3, and 4 (i.e., `[128, 256, 512]`).

If we have only one number on one side of the range, it will include all the remainder of the side that wasn't specified.

- For example, an Upper Bound, `[:5]` will include all elements from the beginning of the sequence up to (but not including) index 5. It is equivalent to `[0, 1, 2, 3, 4]`, assuming zero-based indexing.
- Similarly, a Lower Bound, `[2:]` will include all elements starting from index 2 until the end of the sequence. It will include index 2 and all subsequent indices. For example, if we have a list `[10, 20, 30, 40, 50]`, then `[2:]` will result in `[30, 40, 50]`.

In the provided code snippet, `iloc` is used to select rows and columns based on their positions within the DataFrame. The first argument represents the rows to be selected, and the second argument represents the columns.

- `iloc[:, :-1]`: This notation means selecting all rows and all columns except the last one. It is used to extract all rows and all columns except the last column from the DataFrame.
- `iloc[:, -1]`: This notation means selecting all rows and the last column. It is used to extract all rows and only the last column from the DataFrame.

In most cases, datasets would have all their independent variables ordered from the beginning of each row across all columns, while the dependent variable would mostly occupy the last column within each row.
This means that for most of these cases:

- **X-features** will be extracted by including all rows and columns except for the last one (`iloc[:, :-1]`).
- **y-dependent** is extracted from the last column of each row, excluding all other columns (`iloc[:, -1]`).

#### 3. Split the dataset into training and testing sets

```python
from sklearn.model_selection import train_test_split
...
X_train, X_test, y_test, y_train = train_test_split(X, y)
```

Lastly, once the data is segregated, we can make the final split to extract the training and testing sets - into 4 variables.

Later in the future when we train our model, it would take the `X_train` and `y_train` as inputs, and the prediction would take `X_test` as input.
The result of this prediction can then be compared against the data stored in `y_test` for evaluating the performance and accuracy of our model.

###### Training and Testing Set Split

The `train_test_split` function from the `model_selection` module in scikit-learn is used to split the independent features (X) and the dependent variable (y) into training and testing sets. This is important when we are done training to allow us to test our model with similar, yet unseen, data.

This function takes the X-features and y-dependent variables as input arguments, as well as two additional arguments to better define the split:

- **test_size**: Determines the normalized (0.0 to 1.0) percentage that should be allocated for the testing set (0.0 is 0%, and 1.0 is 100%).
- **random_state**: Introduces a random seed generator for the index selection. This is beneficial for establishing consistency within the randomization.

The function outputs a destructured set of variables corresponding to the train and test data, with each piece conforming to its corresponding split partner. This means that `y_train` would match all the entries in `X_train`, and `y_test` would match all the entries in `X_test`. The destructuring returned from this function must match the following order:

```python
from sklearn.model_selection import train_test_split
...
X_train, X_test, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=0)
```

## Intermediate Data Preprocessing

In some or even most real-world scenarios, our setup for the Preprocessing phase will often need to be more elaborate, in order to have our data well defined, cleaned, and formatted properly so it would match the format expected by the functions for training and predicting.

An intermediate data preprocessing setup would still consist of our previous basic setup as seen earlier, yet it would include some additional steps:

- 1. Importing the Libraries
- 2. Importing the Dataset (+ segregating `Feature X` and `Dependant y` variables)
- 3. Taking Care of Missing Data
- 4. Encoding Categorical Data
  - Encoding the Independent Variable(s)
  - Encoding the Dependant Variable
- 5. Splitting the Dataset into Training Set and Testing Set
- 6. Feature Scaling

This process is a bit more involved, but it does grant us greater control over the Data Preprocessing phase, resulting in better data quality.

For the example of this process, we will be using this tabular data, as stored within the `Data.csv` file:

| Country | Age | Salary | Purchased |
| ------- | --- | ------ | --------- |
| France  | 44  | 72000  | No        |
| Spain   | 27  | 48000  | Yes       |
| Germany | 30  | 54000  | No        |
| Spain   | 38  | 61000  | No        |
| Germany | 40  |        | Yes       |
| France  | 35  | 58000  | Yes       |
| Spain   |     | 52000  | No        |
| France  | 48  | 79000  | Yes       |
| Germany | 50  | 83000  | No        |
| France  | 37  | 67000  | Yes       |

As we can see, this table does have some missing data - for which we will need to account for.

Additionally, some entries are strings - which will need to be converted to numerical values in a way that best describes the correlation of this piece of data within the overall model.

_(Code is also available in `2-DataProcessingTools.py`)_

#### 1. Importing the Libraries
