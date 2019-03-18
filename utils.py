import pandas as pd
import os
import pickle
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import scipy

def drop_columns(df, columns):
    df1 = df.drop(columns, axis='columns')
    return df1

def skewed_features_log1p_transformer(df):
    # Log transform of the skewed numerical features to lessen impact of outliers
    # Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
    # As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
    skewness = df.apply(lambda x: scipy.stats.skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    print(str(skewness.shape[0]) + " skewed numerical features to log transform")
    skewed_features = skewness.index
    df[skewed_features] = np.log1p(df[skewed_features])
    return df


def log1p_transformer(df, column):
    df1 = df.copy()
    df1[column] = np.log1p(df[column])
    return df1

def expm1_transformer(df, column):
    df1 = df.copy()
    df1[column] = np.expm1(df[column])
    return df1


def label_encoder_transformer(df, column):
    """ Encode the column of df with label encoder
    """
    df1 = df.copy()
    model = LabelEncoder()
    result = model.fit_transform(df[column])
    df1[column] = result
    
    return df1


def replace_nan_transformer(df, column, value):
    """ Replace nan of column with value
    """
    df1 = df.copy()
    df1[column] = df[column].fillna(value)
    return df1

def generate_directories(directory):
    
    try:
        os.mkdir(directory)
    except:
        pass

def dump(obj, filename):
    """ dump any object to a pickle filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    
def load(filename):
    """ Load any object from a pickle filename
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_type(s):
    """ Return the type of series item
    """
    return s.dtype

def is_numerical(s):
    """ Returns whether a series is of type numerical"""
    x = get_type(s)
    if x == 'float64' or x == 'int64':
        return True
    return False


def get_numerical_columns(df):
    """ Return the numerical columns of data frame
    """
    columns = []
    for column in df.columns:
        if is_numerical(df[column]):
            columns.append(column)
            
    return columns

def get_columns_with_null(df):
    """ Return columns of data frame with null values
    """
    columns = []
    for column in df.columns:
        if (df[column].isnull().any()):
            columns.append(column)
            
    return columns


def get_numerical_columns_with_null(df):
    """ Return columns of data frame with null values
    """
    columns = []
    for column in df.columns:
        if is_numerical(df[column]):
            if (df[column].isnull().any()):
                columns.append(column)
            
    return columns
            

def get_null_rate(df):
    """ Given the null rate of each numerical columns of df
    """
    columns = dict()
    for column in df.columns:
        if (df[column].isnull().any()):
            columns[column] = df[column].isnull().sum() / len(df)

    columns_ = {'column': [], 'null_rate': []}
    for column, value in columns.items():
        columns_['column'].append(column)
        columns_['null_rate'].append(value)
    return pd.DataFrame(columns_).sort_values(['null_rate'], ascending=False)
    

def get_negative_numbers_ratio(df):
    """ Give the negative numbers ratio of each numerical columns of df
    """
    return df[df>=0].isnull().sum().sum() / df.size


def detect_outliers(df, features, n=0):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    df  dataframe
    features iterable
    n integer
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers


def drop_outliers(df, outliers):
    """ Remove outliers given in iterable outliers
    """
    try:
        df = df.drop(outliers, axis = 0).reset_index(drop=True)
    except KeyError:
        pass
        # transformers_logger.info('{} not found in df'.format(outliers)
    return df
