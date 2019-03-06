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

def transform(type_, df):
    if type_ == 'categorical':
        df1 = pd.DataFrame()
        for column in df.columns:
            if column == 'Embarked': continue
            model = LabelEncoder()
            try:
                result = model.fit_transform(df[column])
                df1[column] = result
            except:
                pass
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
