
#! -----------------------------------------------------------------------------------------------------------------------------------------------------------------
#! 1. Importing the libraries


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#! -----------------------------------------------------------------------------------------------------------------------------------------------------------------
#! 2. Importing Dataset as DataFrame & Segregating the dependant variable from the independent variables

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
In fact, this can result in unwanted bias errors and deviations for our model, generating misinterpreted correlations between the features and the outcome that we
want to predict.
This would cause the model to "understand" that there is some numerical order between these values, and interpret that this order matters - when actually, it doesn't.

In such cases, the Categorical Encoding could be designed to replace the string values with "binary vectors" (OneHotEncoding), populated with the a number of indices 
to match the number of existing categories (unique string values) within that column.
It is like encoding an ID signature within the dataset of each possible category for a particular column. Each option gets a unique ID.
So in the case of `Regime Type: Plutocracy | Technocracy | Democracy`, we would have an array with 3 indices, representing each unique string value, without any
relation or correlation between them:

```python
# Do note, that this effectively redefines the size of the features array - in this case, one column turns into 3 columns

Regime Type
    - Plutocracy    : [1.0, 0.0, 0.0]
    - Technocracy   : [0.0, 1.0, 0.0]
    - Democracy     : [0.0, 0.0, 1.0]

# Using OneHotEncoding, we can definitely say that none of these counts "more" or "less" than the others.
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
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4.1 Encoding The Feature Matrix
"""
Our example dataset, 'Data.csv', can be used with OneHotEncoding to encode the `Country` column, which has only 3 possible categories: Spain, France, and Germany.
This will resize and reshape the X features array, causing this one column to extend across 3 new columns, while forcing the unaffected following columns to 
relocate - but the upside is that our model would perform better with less bias.

One Hot Encoding consists of creating binary vectors which would represent each category in the column.
So for example, instead of this array:
`
#Pre-Encoding
[
    [Spain, 42, 55000, Yes],
    [France, 45, 57500, No],
    [Germany, 30, 49000, Yes]
]`
We get this modified array:
`
#Post One Hot Encoding
[
    [1, 0, 0, 42, 55000, Yes],
    [0, 1, 0, 45, 57500, No],
    [0, 0, 1, 30, 49000, Yes]
]`
"""
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
"""
To Hot Encode our string categorical column, two classes from **SciKit-Learn** must be imported:
    - the `ColumnTransformer` class from the `compose` module
    - the `OneHotEncoder` class from the `preprocessing` module

This process involves the following steps:
    1) Instantiate a `ColumnTransformer` object, and provide 2 arguments for the constructor to define the encoding and transformation settings;
        1.1) First argument - `transformers` - expects a list of tuples with 3 values, formatted as a parentheses in square brackets `[()]`
            1.1.1) 0: a string that serves as an identifier for the encoder
            1.1.2) 1: an instance of the encoder class we want to use
            1.1.3) 2: the column index/range on which the encoding should be applied
        1.2) Second argument - `remainder` - defines how the unaffected columns should be treated - expects 3 possible arguments
            1.2.1) 'drop' - This is the default value. It means that any remaining columns that were not transformed will be dropped from the 
                            output of the ColumnTransformer. In other words, only the transformed columns will be included in the resulting dataset.
            1.2.2) 'passthrough' - This argument indicates that any remaining columns that were not transformed should be included in the output without any 
                                   changes. These columns are "passed through" the transformer pipeline as they are. This is useful when you want to keep some 
                                   columns unchanged while applying transformations to others.
            1.2.3)  Transformer instance - You can also pass a transformer instance to the remainder parameter. This transformer will be applied to the remaining 
                                           columns. For example, you can pass an instance of StandardScaler or MinMaxScaler to apply feature scaling to 
                                           the remaining columns.
                                           
    2) Use the newly defined `ColumnTransformer` object to encode, fit and apply the transformation of the resulting encoded data to our X features array.
       For this purpose, we use the `fit_transform()` method of the `ColumnTransformer` object, which takes as input an array, and returns the transformed version
       of it, with the encoding applied to it.
        2.1) This requires us to set the returned value from this function back to the same input array - `X = ct.fit_transform(X)`
        2.2) However, the array returned from this function is formatted differently than how it had originally entered it, and it will not be useable later for
             the learning stage in the future.
             For this reason, the array this function returns to us must be converted into a format that would match our future needs, which is, a numpy array -
             `X = np.array(ct.fit_transform(X))`
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4.2 Encoding the Dependent Variable

"""
Sometimes we have string category columns that represent scalar values or binary data, which for in this case it indeed makes sense to use numerically ordered
ranges, such as 0 to 1, 0 to 100, -1 to +1 - the range we choose should dependant on the purpose that the category represents.

In our `Data.csv` dataset, the last column - which is also our predicted `y` variable - is a string column that represents binary data, of 
a true/false boolean value.
In this case, we can most logically encode these two possible outcomes as 0 and 1, a range that represents one possible choice among only 2 options.
"""
# from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

"""
For the purpose of a binary encoding operation, we can use the `LabelEncoder` class from the **Scikit-Learn** `preprocessing` module.

This process involves two simple steps:
    1) Instantiating a `LabelEncoder` object
    2) Encode the data, fit and apply to the desired array
        - NOTE - This array does not need to be converted to a numpy array, as previously seen with the `OneHotEncoder`, since the FitTransform method of the
                 `LabelEncoder` takes in and returns a 1-dimensional array.
"""

#! -----------------------------------------------------------------------------------------------------------------------------------------------------------------
#! 5. Splitting the Data Into Train Sets & Test Sets

# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

"""
The `model_selection` module from the **scikit-learn** library contains the `train_test_split` function, which is used to split our independent features (X) and the 
dependant variable (y) into training sets and testing sets.
This is important for later when we are done training, to allow ourselves to test our model with similar yet unseen data.
Also, this is the format expected by most ML models that we will implement in the future.

This function takes the X-features and y-dependant variables as input arguments, as well as 2 additional arguments to better define the split:
- **test_size** - Determines the normalized (0.0 to 1.0) percentage that should be allocated for the testing set - 0.0 is 0%, 1.0 is 100%
- **random_state** - Introduces a random seed generator for the indices selection - this is beneficial for establishing consistency within the randomization

The function outputs a destructured set of variables, corresponding to the train and test data, with each piece conforming to it's corresponding split partner.
This means that `y_train` would match all the entries at `X_train`, and `y_test` would match all the entries at `X_test`.
The destructuring returned from this function must match the following order:

`X_train, X_test, y_test, y_train = train_test_split(...)`
"""

#! -----------------------------------------------------------------------------------------------------------------------------------------------------------------
#! 6. Feature Scaling

"""
- - - - - - - - - - - - - - - - - - - - - - - - - - 

Feature Scaling is a procedure performed for the purpose of preventing one feature from dominating other features. It takes the Mean and the Standard Deviation 
of numerical columns (scalar values only).
The choice whether to apply Feature Scaling or not would depend on many factors, such as the dataset used, the numerical differences within AND across columns, the 
type of machine learning model we are going to use, and more.
Some cases may not require feature scaling at all, while other cases may demand heavy scaling of the features.

- - - - - - - - - - - - - - - - - - - - - - - - - - 

When a feature is dominated by an other feature, the dominated feature won't even be considered by the machine learning model, due to the dominating feature 
overpowering it, rendering it out of importance.
This can often occur because of extremely large numbers and/or large differences between numbers.

Feature Scaling must be performed after the split process of train/test sets, because otherwise the scaled values of the sets would be compromised by one another.
To put it more simply, if feature scaling is applied before the "split", then the `y` data which should be not be known to our model before we make a prediction, would
influence the ranges by which the `X` data is scaled - and vice versa.
Feature Scaling takes the Mean and the Standard Deviation of numerical columns to perform the scaling, so if we apply the scaling before the split, it would get the
Mean and the Standard Deviation of all the values in the column - including the ones that would later be split into the test sets.
This result is a negative, unwanted result, which earned a notorious title called Data Leak or Information Leakage.
So for this reason, Feature Scaling must be applied after the "split".

Later, once our model is trained, we would scale 'X_test' with the same mean and standard deviation produced by fitting 'X_train'.
Meaning, something like: 
                    ```
                        model.fit(X_train)
                        model.transform(X_train)
                        model.transform(X_test)
                    ```

- - - - - - - - - - - - - - - - - - - - - - - - - - 

There are two common methods of feature scaling:
    1. Standardization (Z-score normalization):
       This method transforms the features so that they have a mean of zero and a standard deviation of one. It subtracts the mean value of the feature 
       from each data point and divides by the standard deviation. The formula for standardization is:  

        ```
        X_scaled = (X - mean(X)) / standard_deviation(X)
        ```

        Standardization makes the data centered around zero, with a standard deviation of one.

    2. Normalization (Min-Max scaling):

       This method scales the features to a specific range, typically between 0 and 1. It subtracts the minimum value of the feature from each data point 
       and divides by the range (maximum value minus minimum value). The formula for normalization is:

       ```
       X_scaled = (X - min(X)) / (max(X) - min(X))
       ```

       Normalization maps the data to a fixed range, preserving the relative relationships between the values.

Both standardization and normalization are effective techniques for feature scaling, and the choice between them depends on the specific requirements 
of your problem and the characteristics of your data.
"""

# from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[:, 3:] = scaler.fit_transform(X_train[:, 3:])
X_test[:, 3:] = scaler.transform(X_test[:, 3:])

"""
For our example purposes, we use the Standardization technique to scale the numerical columns of our `X_train` dataset chunk.
This can be broken down to 3 simple steps:
    1) Import the `StandardScaler` class from the **Scikit-Learn** `preprocessing` module, and instantiate a new object
    2) Apply the `fit_transform()` method to the target matrix on the desired range of columns that are ought to be scaled
        - `X_train[:, rangeStart.location : rangeEnd.location] = scaler.fit_transform(X_train[:, rangeStart.location : rangeEnd.location])`
    3) Finally, we apply the same Transformation of the same Scaler that was fitted for the Training Data - onto the Testing Data
        - `X_test[:, rangeStart.location : rangeEnd.location] = scaler.transform(X_test[:, rangeStart.location : rangeEnd.location])`

*Regarding step #3:
In order to make predictions that will be congruent with the way the model was trained, we need to apply the same scaler that was used on the training set 
onto the test set, so that we can get indeed the same transformation of the "Biased Perception" that was formed during the training, and sustain that "Biased 
Perception" while looking at new, unseen data.

- - - - - - - - - - - - - - - - - - - - - - - - - - 

Do note, that the dummy variables we encode earlier in the preprocessing process don't need to be scaled - and as a general rule - pre-scaled values do not need to 
be included in the Feature Scaling.
For this reason, we exclude any encoded dummy variables, pre-scaled variables or any undesired columns from the feature scaling.
Since our X matrix should look like this at this point;
`   X_train = [
        [0.0 1.0 0.0 30.0 54000.0]
        [0.0 0.0 1.0 38.0 61000.0]
        [0.0 1.0 0.0 40.0 63777.7]
        [1.0 0.0 0.0 35.0 58000.0]
        [0.0 0.0 1.0 38.78 52000.0]
        [1.0 0.0 0.0 48.0 79000.0]
        [0.0 1.0 0.0 50.0 83000.0]
        [1.0 0.0 0.0 37.0 67000.0]
    ]
`
The exclusion of the undesired columns is defined in such a way - `X_train[:, 3:]`.

This simplicity and easily modifiable solution to this exclusion, as well as the simple and reusable segregation of the `Predictor (X)` from `Predictable(y)` that we
performed earlier, is why it is important for to conform to the following format when initially creating a dataset: 
    
    | [...String_Columns] | [...Number_Columns] | Predictable_COLUMN |

This way, the preprocessing stage would be much simpler, as well as we would benefit a tidy dataset.
"""

#! -----------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
This concludes all the basic tools and concepts of Data Preprocessing we should be aware of when first starting out with ML!
"""
print("Training Data:")
print(X_train)
print("Testing Data:")
print(X_test)
