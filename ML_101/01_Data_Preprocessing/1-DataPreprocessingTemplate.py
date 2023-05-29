# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. import the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

"""
These libraries would cover most of our data processing/preprocessing manipulations and plots - and would make up for most of our basic ML tech stack:

*NumPy - Matrices & Math 
NumPy is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, 
along with a collection of mathematical functions to operate on these arrays efficiently. NumPy is widely used in data analysis, numerical computations, and 
machine learning due to its fast array processing capabilities.

*Matplotlib - Visualize Data
Matplotlib is a plotting library for creating static, animated, and interactive visualizations in Python. It provides a wide range 
of customizable plots, including line plots, scatter plots, bar plots, histograms, and more. Matplotlib is commonly used for data visualization tasks, enabling 
users to create clear and informative plots to explore and present their data.

*Pandas - Manipulate Data 
Pandas is a powerful data manipulation and analysis library for Python. It provides easy-to-use data structures, such as DataFrames, for efficiently 
handling structured data. With Pandas, you can load, manipulate, filter, and transform data, perform operations like merging and grouping, handle missing 
data, and more. It is widely used for data preprocessing, data cleaning, exploratory data analysis, and data wrangling tasks.

*Scikit-Learn - Data & ML Ops
Scikit-learn, often abbreviated as sklearn, is a popular machine learning library in Python. It provides a wide range of machine learning algorithms 
and tools for tasks such as classification, regression, clustering, dimensionality reduction, model selection, and evaluation. Scikit-learn is known for 
its user-friendly interface and extensive documentation, making it accessible to both beginners and experienced practitioners. 
It also integrates well with other scientific libraries, such as NumPy and Pandas, for seamless data processing and model building pipelines.

"""

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Import the Dataset - and segregate the dependant variable from the independent variables

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""
The `iloc` method from pandas dataset object (DataFrame object) is an extremely flexible function that can take index members or even index ranges as
arguments, to easily filter a list of entries in 2-Dimensional tabular data.
This method allows us to select rows and columns in a DataFrame using integer-based indexing.

ILOC Arguments:
1) First argument specifies the first dimension of the array, the row
2) Second argument specifies the second dimension of the array, the column

The arguments passed to this function can be index numbers, but it can even be Python Ranges.
A range is denoted by a colon `:`, which by itself describes the entire range.
- The number on the left of the colon represents the starting point (inclusive) of the range.
- The number on the right of the colon represents the ending point (exclusive) of the range.

For example, 
`
my_list = [32, 64, 128, 256, 512, 1024, 2048]
my_list[2:5]
`
this will extract elements from `my_list` starting at index 2 (inclusive) and ending at index 5 (exclusive). 
The resulting range will include elements at indices 2, 3, and 4 - in this case, `[128, 256, 512]`.

If we have only one number on one side of the range, it will include all the remainder of the side that wasn't specified.
    - For example, `[:5]` will include all elements from the beginning of the sequence up to (but not including) index 5. 
      It is equivalent to `[0, 1, 2, 3, 4]`, assuming a zero-based indexing.
    - Similarly, `[2:]` will include all elements starting from index 2 until the end of the sequence. It will include 
      index 2 and all subsequent indices. For example, if we have a list `[10, 20, 30, 40, 50]`, then `[2:]` will result in `[30, 40, 50]`.

When you use the range [:-1], it selects all the elements from the beginning of the sequence up to the last element excluding the last element itself.
In the provided code snippet, `iloc` is used to select rows and columns based on their positions within the DataFrame. The first argument represents the rows 
to be selected, and the second argument represents the columns.
    - **[:, :-1]**: This notation means selecting all rows (:) and all columns except the last one (:-1). It is used to extract all rows and all 
      columns except the last column from the DataFrame.
    - **[:, -1]**: This notation means selecting all rows (:) and the last column (-1). It is used to extract all rows and only the last column from the DataFrame.

In most cases, datasets would have all of their independent variables ordered from the beginning of each row across all columns - while the dependant variable
would mostly occupy the last column within each row.
This means that for most of these cases; 
- **X-features** will be extracted by including all rows and columns except for the last one - `iloc[:, :-1]` 
- **y-dependant** is extracted from the last column of each row, excluding all other columns - `iloc[:, -1]`

"""

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Split the dataset to Training and Testing sets

X_train, X_test, y_test, y_train = train_test_split(
    X, y, test_size=0.2, random_state=0)

"""
The `model_selection` module from the **scikit-learn** library contains the `train_test_split` function, which is used to split our independent features (X) and the 
dependant variable (y) into training sets and testing sets.
This is important for later when we are done training, to allow ourselves to test our model with similar yet unseen data.

This function takes the X-features and y-dependant variables as input arguments, as well as 2 additional arguments to better define the split:
- **test_size** - Determines the normalized (0.0 to 1.0) percentage that should be allocated for the testing set - 0.0 is 0%, 1.0 is 100%
- **random_state** - Introduces a random seed generator for the indices selection - this is beneficial for establishing consistency within the randomization

The function outputs a destructured set of variables, corresponding to the train and test data, with each piece conforming to it's corresponding split partner.
This means that `y_train` would match all the entries at `X_train`, and `y_test` would match all the entries at `X_test`.
The destructuring returned from this function must match the following order:

`X_train, X_test, y_test, y_train = train_test_split(...)`

"""

print(X_train)
print(y_train)
print(X_test)
print(y_test)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
This sums a very basic data preprocessing template setup, which could be used and expanded upon as we may need.
However, for some situations, we may find ourselves needing to take some extra steps, such as cleaning the data, accounting for missing data, perform Feature Scaling
to standardize or normalize the data for better accuracy, and more.
"""
