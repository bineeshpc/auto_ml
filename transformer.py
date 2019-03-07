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
from sklearn.preprocessing import RobustScaler

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

def f(x):
    d = dict(Mlle='Miss',
            Ms='Miss',
            Mme='Mrs')
    
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
    
                
               

def transform(type_, df):
    if type_ == 'categorical':
        df1 = pd.DataFrame()
        for column in df.columns:
            if column == 'Embarked':
                # s is most frequent
                # so replace null with s
                df[column].replace(np.nan, 'S', inplace=True)
            model = LabelEncoder()
            try:
                result = model.fit_transform(df[column])
                df1[column] = result
            except:
                print(column)
                pass
        return df1
    if type_ == 'text':
        df1 = pd.DataFrame()
        for column in df.columns:
            if column == 'Name':
                dummy = df[column].apply(f)
        model = LabelEncoder()
        result = model.fit_transform(dummy)
        df1['Title'] = result
        return df1
    if type_ == 'numerical':
        df1 = pd.DataFrame()
        for column in df.columns:
            if column == 'Age':
                directory = config_parser.get_configuration(args.configfile).get_directory('transformer')
                df2 = pd.read_csv(os.path.join(directory, 'text.csv'))
                df3 = combine_df_same_length(df, df2)
                null_values_df = df3[df3[column].isnull()]
                good_values_df = df3[df3[column].isnull() != True]
                # print(null_values_df.shape)
                # print(good_values_df.shape)
                # print(df3.shape)
                good_values_df['Fare'].fillna(df['Fare'].median(), inplace=True)
                # print(good_values_df.columns, column)
                X_train = good_values_df.drop(column, axis='columns').values
                y_train = good_values_df[column].values.reshape(-1, 1)
                # print(X_train.shape)
                # print(y_train.shape)
                steps = [('scaler', RobustScaler()),
                         ('reg', LinearRegression())
                         ]
                pipeline = Pipeline(steps)
                model = pipeline.fit(X_train, y_train)
                X_test = null_values_df.drop(column, axis='columns').values
                
                y_pred = model.predict(X_test)

                filled_values_df = null_values_df.copy()
                filled_values_df[column] = y_pred
                df4 = recreate_df(df3, good_values_df, filled_values_df, 'PassengerId')
                df1[column] = df4[column]
                

            elif column == 'Fare':
                df1[column] = df[column].fillna(df[column].median())      
            else:
                df1[column] = df[column]

        return df1
    else:
        return df

    
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
    for type_, df in all_types.items():
            filename = '{}.csv'.format(type_)
            filename = os.path.join(directory, filename)
            transformed_df = transform(type_, df)
            transformed_df.to_csv(filename, index=False)
    

if __name__ == "__main__":
    args = parse_cmdline()
    
    transformers_logger = logger.get_logger('transformer',
                                            'transformer',
                                            args.configfile)

    main(args)
