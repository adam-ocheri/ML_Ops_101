
#! -----------------------------------------------------------------------------------------------------------------------------------------------------------------
#! 1. import the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#! -----------------------------------------------------------------------------------------------------------------------------------------------------------------
#! 2. Import Data as DataFrame & Segregate the dependant variable from the independent variables

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""
We always need to separate the independent variables that use, shape and influence the model's internal statistical calculations - from the dependant 
variable that we want to predict.

For this, we use the DataFrame function "iloc", which is locating indices according to the range arguments we provide:
`...iloc[row, column]...`

By specifying ranges using index numbers or range operators, we can

For our example, we will be using the `Data.csv` file stored in this folder:

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

- 'Country', 'Age', and 'Salary' are the independent Feature X variables    - The Matrix of features
- 'Purchased' is the dependant piece of data that we want to predict        - The Dependant Variable Vector

"""

#! -----------------------------------------------------------------------------------------------------------------------------------------------------------------
#! 3. Taking Care of Missing Data

"""
Sometimes, some rows within the dataset we use may have some of their values missing.
In the case of our `Data.csv` file, we can clearly see that some values are missing at 
the `[:, 1]` and `[:, 2]` columns (somewhere within the second and third columns of the dataset).

One option we can go by, if our dataset is large enough, is to simply delete all of the rows which are lacking data.
This may be the best solution, as we get to avoid any inconsistencies in our dataset altogether.

However, this option of deleting the "infected" rows may not bode well if our dataset is too small to afford discarding any amount of data whatsoever.
In such a case, we would have to supply some substitute values for the missing bits of data in this dataset.

For this, we can use the `SimpleImputer` class from the `sklearn.impute` module.
"""

# from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

"""
The `SimpleImputer` class is used to define an imputer object that would store the settings by which the imputation should occur.

The `SimpleImputer` can help with replacing any missing value with the average or the median of all the existing values within that particular column.

**SimpleImputer** constructor arguments:

    - `missing_values` - The missing values to search for and impute. 
       using `np.nan` in this example to search for NaN values, where a numerical value is missing from a column within the dataset

    - `strategy` - The mathematical equation to apply for all entries of a single column, to arrive at the desired mitigated value.
        - Possible arguments:
            - 'mean': This strategy replaces missing values with the mean of each feature along the specified axis. The mean is the average value calculated by summing up all the values and dividing 
               by the number of non-missing values. It is a commonly used imputation strategy for numerical data.
            - 'median': This strategy replaces missing values with the median of each feature along the specified axis. The median is the middle value in a sorted list of values.
            - 'most_frequent': This strategy replaces missing values with the most frequent value (mode) of each feature along the specified axis. It is suitable for imputing categorical or discrete data.
            - 'constant': This strategy replaces missing values with a constant value provided by the fill_value parameter. It allows you to specify a specific value to fill in the missing entries.

When using the 'mean' strategy with `SimpleImputer`, missing values in each feature will be replaced with the mean value of that feature. 
This approach assumes that the missing values are missing at random and that the mean provides a reasonable estimate of the missing values. 
It is important to note that using the mean strategy can introduce bias in the data if the missing values are not missing at random 
or if there are outliers present.

By applying the 'mean' imputation strategy, you can ensure that missing values are replaced with a representative value based on the available data, 
allowing you to perform further analysis or modeling on a complete dataset.
"""

imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

"""
Once the imputer object is defined, we still need to use it for replacing the missing data with some actual value(s).
This procedure involves 2 operations:
1. Fitting the data ([ro::ws, col:u:mns]) that should be processed by the `SimpleImputer`, using the `fit()` method
    - Can only accept columns of numerical values; No strings allowed
2. Transforming that data, and applying that transformation to our dataset, using the `transform()` method
    - No strings allowed; Most commonly fed with the same input data that was provided for the fitting method

*`fit()` Method:
In the case of our `Data.csv` file, only the `[:, 1]` and `[:, 2]` indices are columns that represent numerical value.
For this reason, the fitting of the imputer would be `[:, 1:3]` 
`imputer.fit(X[:, 1:3])`
(remember that the Upper Bound is not inclusive, so the range of `1:3` does include the column at 
index `[1]` and the column at index `[2]`, effectively excluding the string columns).

*`transform()` Method:
The transform method should take in the same input data that was fed to the fit method - which is again, the range of numerical-type columns of all 
independent variables in the dataset.
Then, the transform method returns the new updated version of the feature matrix `X`, containing the generated data values for 
whatever missing entries (under numerical-type columns) that the dataset was lacking.
Finally, the returned object needs to be assigned to the feature matrix `X` within the same range on which the imputation acted upon.
`X[:, 1:3] = imputer.transform(X[:, 1:3])`
In the case of our `Data.csv` file, only the `[:, 1]` and `[:, 2]` indices are columns that represent numerical value.
These are also the indices that were fitted for in the fitting operation.
So for this reason, the transformation of the imputer would again be `[:, 1:3]`, as well as the range to which this transformation applies to.

If we were to work with a very large dataset, it might be hard to pinpoint exactly which columns are missing data entries.
So as a general rule of thumb, we should most always impute all of the columns representing numerical types within the dataset 
(or estimate how much in % out of all training data the "infected" data takes up - and if it is a small percentage 
within a large dataset - we may simply delete these "infected" rows altogether).

"""

#! -----------------------------------------------------------------------------------------------------------------------------------------------------------------
#! 4. Encoding Categorical Data

"""
Oftentimes, a dataset may contain one or even several string columns in it.
This can be a problem, if our model isn't built to support string data or even make sense of it.

For these kind of situations, we will need to utilize Categorical Data Encoding.
The basic idea is to simply encode and replace any string data with a meaningful numeric representation of that data.
However, the purpose of the string-type-column must be taken into account when choosing a numerical format to encode that string data to.

To put it more simply, for example, if we had a column for `Regime Type` with all entries consisting only of "Plutocracy | Technocracy | Democracy", then it 
would not make much sense to encode them from 0 to 2 - because these numbers represent a range which implies some incrementing/decrementing spectrum-like relation 
between these values.
In fact, this can result in unwanted bias errors and deviations for our model, which is something we definitely want to avoid.

In such cases, the Categorical Encoding could be designed to replace the string values with "dummy arrays", populated with the a number of indices to match 
the number of existing unique string values within that column.
So in the case of `Regime Type: Plutocracy | Technocracy | Democracy`, we would have an array with 3 indices, representing each unique string value, without any
relation or correlation between them:

```python
Regime Type
    - Plutocracy    : [1.0, 0.0, 0.0]
    - Technocracy   : [0.0, 1.0, 0.0]
    - Democracy     : [0.0, 0.0, 1.0]

# This way, we can definitely say that none of these counts "more" or "less" than the others.
# TL;DR: They all equally suck!
```

Conversely, if we have a string column that represents some true/false boolean values, then we can very well encode it within a 0 to 1 range - since these 2 strings
values DOES represent some relation between them - and also, a boolean type is fundamentally a 0 or 1 value, so in such cases, relative numbers DOES make sense:

```python
Regime Type
    - Worse One    : 0
    - Better One   : 1

# This way, we can definitely say that one of these counts "more" or "less" than the other.
# TL;DR: Each sucks within a context of some scalar, measurable range!
```
"""
