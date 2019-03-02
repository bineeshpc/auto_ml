#! /usr/bin/env python

"""
Given a data frame split it into holdout set and the working data

Hold out set will never be seen by the models.
Hold out set will be used for evaluation

"""

from sklearn.model_selection import train_test_split
import logger
import config_parser
import utils
import argparse
import pandas as pd
import numpy as np
import os

splitter_logger = logger.get_logger('splitter',
'splitter' )

def parse_cmdline():
    parser = argparse.ArgumentParser(description='exploratory data analysis in python')
    parser.add_argument('inputfile',
                        type=str,                    
                        help='Inputfile')
    parser.add_argument('y_column',
                        type=str,                    
                        help='y column value to predict')
    
    args = parser.parse_args()
    return args


def split_(df, y_column):
    df1 = df.drop(y_column, axis='columns')
    y_column_df = pd.DataFrame({y_column: df[y_column]})
    try:
        X_train, X_test, y_train, y_test = train_test_split(df1, 
        y_column_df,
         test_size=.2,
        stratify=y_column_df)
    except ValueError:
        print('trying without stratify')
        X_train, X_test, y_train, y_test = train_test_split(df1, 
        y_column_df,
         test_size=.2
        )

    return X_train, X_test, y_train, y_test

def main(args):
    if args.inputfile:
        df = pd.read_csv(args.inputfile)
        X_train, X_test, y_train, y_test = split_(df, args.y_column)
        dfs = X_train, X_test, y_train, y_test
        names = ['X_train', 'X_test', 'y_train', 'y_test']

        configuration = config_parser.configuration
        directory = configuration.get_directory('splitter')
        
        for name, df in zip(names, dfs):
            filename = os.path.join(directory, name + '.csv')
            df.to_csv(filename, index=False)
            


if __name__ == "__main__":
    args = parse_cmdline()
    main(args)
