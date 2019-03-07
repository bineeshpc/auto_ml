#! /usr/bin/env python

"""
Given a data frame split the data based on data type
Known data types are

Numerical
Categorical
Text

"""
import logger
import config_parser
import utils
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler



def parse_cmdline():
    parser = argparse.ArgumentParser(description='Input directory')
    parser.add_argument('bool_',
                        type=str,                    
                        help='bool type data')
    parser.add_argument('categorical',
                        type=str,                    
                        help='categorical type data')
    parser.add_argument('text',
                        type=str,                    
                        help='text type data')
    parser.add_argument('numerical',
                        type=str,                    
                        help='numerical type data')
    parser.add_argument('configfile',
                        type=str,                    
                        help='configfile')

    args = parser.parse_args()
    return args


class Transformer:
    def __init__(self):
        """ Transformer is a 4 tuple
        tuple values can be empty
        
        """
        self.transformer = None

    def do_transformation(self, name, function, args, kwargs):
        """ Add a transformer 
        (a string representing name of transformer function,
        transformer function,
        args is a tuple or list that can be used a positional function arguments,
        kwargs is a dictionary for key word functional arguments)
        """
        transformers_logger.debug("Adding {} {} {} {}".format(name, function, args, kwargs))
        transformers_logger.info('Adding {} {}'.format(name, function))
        self.transformer = (name, function, args, kwargs)
        return self.execute(function, args, kwargs)
        
    def execute(self,function, args, kwargs):
        transformers_logger.debug("Executing {} {} {}".format(function, args, kwargs))
        return function(*args, **kwargs)



def extract_title(x):
    d = dict(Mlle='Miss',
            Ms='Miss',
            Mme='Mrs')
    for i in ['Lady', 'the Countess','Countess','Capt',
     'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
     d[i] = 'Rare'

    a = x.split(',')[1].split('.')[0].strip()
    if a in d.keys():
        a = d[a]
    return a

def combine_df_same_length(df1, df2):
    length = len(df2)
    d = dict([(i, [])for i in df1.columns])
    for column_ in df2.columns:
        d[column_] = []
    for i in range(length):
        for column_ in df1.columns:
            d[column_].append(df1[column_][i])
        for column_ in df2.columns:
            d[column_].append(df2[column_][i])
    df3 = pd.DataFrame(d)
    return df3
    

def recreate_df(df3, good_values_df, filled_values_df, c_id):
    x = df3[c_id] == df3.index + 1
    df4 = df3.set_index(c_id)
    df5 = good_values_df.set_index(c_id)
    df6 = filled_values_df.set_index(c_id)
    df7 = pd.concat([df5['Age'], df5['Age']])
    #df8 = df3['Age'] = df7
    df7.index = range(len(df7))
    # print(df7.index)
    # print(df3.index)
    df3['Age'] = df7
    return df3
    
                
               

def replace_nan_transformer(df, column, value):
    """ Replace nan of column with value
    """
    df1 = df.copy()
    df1[column] = df[column].fillna(value)
            
    return df1

def label_encoder_transformer(df, column):
    """ Encode the column of df with label encoder
    """
    df1 = df.copy()
    model = LabelEncoder()
    result = model.fit_transform(df[column])
    df1[column] = result
    return df1

def function_apply_transformer(df, function, column, new_column):
    """ Apply function to df on column delete the column and 
    the result will be named as a new_column 
    """
    df1 = df.copy()
    extracted = df[column].apply(function)
    df1.drop(column, axis='columns')
    df1[new_column] = extracted
    return df1

def drop_columns(df, columns):
    return df.drop(columns, axis='columns')
    


def fill_column(df, text_df, column, scaler, regressor):
    """ column of df is predicted using text_df and a regressor
    """
    df1 = df.copy()
    df3 = combine_df_same_length(df, text_df)
    null_values_df = df3[df3[column].isnull()]
    good_values_df = df3[df3[column].isnull() != True]
    # print(null_values_df.shape)
    # print(good_values_df.shape)
    # print(df3.shape)
    # print(good_values_df.columns, column)
    
    X_train = good_values_df.drop(column, axis='columns').values
    y_train = good_values_df[column].values.reshape(-1, 1)
    # print(X_train.shape)
    # print(y_train.shape)
    steps = [('scaler', scaler),
                ('regressor', regressor)
            ]
    pipeline = Pipeline(steps)
    model = pipeline.fit(X_train, y_train)
    X_test = null_values_df.drop(column, axis='columns').values
    y_pred = model.predict(X_test)

    filled_values_df = null_values_df.copy()
    filled_values_df[column] = y_pred
    df4 = recreate_df(df3, good_values_df, filled_values_df, 'PassengerId')
    df1[column] = df4[column]
    return df1
    
def ticket_modifier(x):
    y = x.split()
    if len(y) > 1:
        return y[0].replace('/', '').replace('.', '').strip()
    else:
        return 'X'

def create_family_size(df, column_formed, column_1, column_2):
    df[column_formed] = df[column_1] + df[column_2] + 1
    return df

def create_family_features(dataset):
    dataset['Single'] = dataset['Family_Size'].map(lambda s: 1 if s == 1 else 0)
    dataset['Small_Family'] = dataset['Family_Size'].map(lambda s: 1 if  s == 2  else 0)
    dataset['Medium_Family'] = dataset['Family_Size'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['Large_Family'] = dataset['Family_Size'].map(lambda s: 1 if s >= 5 else 0)
    return dataset

def main(args):
    def read_csv(location):
        try:
            df = pd.read_csv(location)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        return df
            
    bool_ = read_csv(args.bool_)
    categorical = read_csv(args.categorical)
    text = read_csv(args.text)
    numerical = read_csv(args.numerical)

    type_s = ['bool_', 'categorical', 'text', 'numerical']
    dfs = [bool_, categorical, text, numerical]

    all_types = dict(zip(type_s, dfs))
    directory = config_parser.get_configuration(args.configfile).get_directory('transformer')
    transformers_logger.info(directory)
    for type_, df in all_types.items():
        transformer = Transformer()
        filename = '{}.csv'.format(type_)
        filename = os.path.join(directory, filename)
        transformers_logger.info(type_)
        if type_ == 'bool_':
            df = transformer.do_transformation('identity transformer', (lambda x: x), (df,), {})
        elif type_ == 'categorical':
            df = transformer.do_transformation('replace nan of embarked column', replace_nan_transformer, 
            (df, 'Embarked', df['Embarked'].mode()[0]), {})
            for column_ in df.columns:
                df = transformer.do_transformation('label encode transformer', label_encoder_transformer, 
            (df, column_), {})
        elif type_ == 'text':
            df = transformer.do_transformation('extract title transformer', function_apply_transformer,
            (df, extract_title, 'Name', 'Title'),
            {})
            df = transformer.do_transformation('label encode transformer', label_encoder_transformer, 
            (df, 'Title'), {})
            
            df = transformer.do_transformation('Extract ticket info', 
            function_apply_transformer, (df, ticket_modifier, 'Ticket', 'Ticket_Modified'), {})
            
            # dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
            df = transformer.do_transformation('replace nan of cabin column', replace_nan_transformer, 
            (df, 'Cabin', 'X'), {})

            df = transformer.do_transformation('Extract Cabin info first letter', 
            function_apply_transformer, (df, (lambda x: x[0]), 'Cabin', 'Cabin_Modified'), {})
            
            df = transformer.do_transformation('label encode transformer', label_encoder_transformer, 
            (df, 'Cabin_Modified'), {})
            df = transformer.do_transformation('label encode transformer', label_encoder_transformer, 
            (df, 'Ticket_Modified'), {})
            

            df = transformer.do_transformation('drop columns', drop_columns,
            (df, ['Name', 'Ticket', 'Cabin']),
            {})
            
        elif type_ == 'numerical':
            df = transformer.do_transformation('replace nan of fare column', replace_nan_transformer, 
            (df, 'Fare', df['Fare'].median()), {})
            # need to use the generated column title to fill/complete the missing entries in age
            text_df = pd.read_csv(os.path.join(directory, 'text.csv'))
            df = transformer.do_transformation('fill missing entries in age', fill_column, 
            (df, text_df, 'Age', StandardScaler(), LinearRegression()),
            {}
            )
            df = transformer.do_transformation('Create family size feature', create_family_size, 
            (df, 'Family_Size', 'Parch', 'SibSp'),
            {}
            )
            df = transformer.do_transformation('Create family size special features', create_family_features, 
            (df, ),
            {}
            )               

            df = transformer.do_transformation('drop columns', drop_columns,
            (df, ['PassengerId']),
            {})
        df.to_csv(filename, index=False)


if __name__ == "__main__":
    args = parse_cmdline()
    
    # global transformers_logger
    transformers_logger = logger.get_logger('transformer',
                                            'transformer',
                                            args.configfile)

    main(args)
