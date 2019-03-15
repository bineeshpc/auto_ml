import pandas as pd
import os
import pickle


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
    