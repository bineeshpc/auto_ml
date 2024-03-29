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

def parse_cmdline():
    parser = argparse.ArgumentParser(description='Generate data types')
    parser.add_argument('inputfile',
                        type=str,                    
                        help='Inputfile')
    parser.add_argument('configfile',
                        type=str,                    
                        help='Configfile')
    
    args = parser.parse_args()
    return args


def get_data_types(df, configfile):
    numerical = []
    categorical = []
    text = []
    bool_ = []
    categorical_columns = config_parser.get_configuration(configfile).get_attribute('generate_datatypes', 'categorical')
    for column in df.columns:
        dtype = df[column].dtype
        if  dtype == 'float64' or dtype == 'int64':
            numerical.append(column)
        elif column in categorical_columns:
            categorical.append(column)
        elif dtype == 'bool':
            bool_.append(column)
        elif dtype == 'object':
            text.append(column)
        
    values = ['numerical',
    'categorical',
    'text',
    'bool_'
    ]
    zipped = zip(values, [numerical, categorical, text, bool_])
    return dict(zipped)

def generate_data_types(inputfile, configfile):
    df = pd.read_csv(inputfile)
    
    configuration = config_parser.get_configuration(configfile)
    dfs = dict()
    directory = configuration.get_directory('generate_datatypes')
    for type_, columns in get_data_types(df, configfile).items():
        dfs[type_] = df[columns]
        filename = os.path.join(directory, type_ + '.csv')
        if dfs[type_].empty:
            with open(filename, 'w') as f:
                pass
        else:
            dfs[type_].to_csv(filename, index=False)
    return dfs


def main(args):
    
    generate_datatypes_logger = logger.get_logger('generate_data_types',
                                                  'generate_datatypes',
                                                  args.configfile
    )
    generate_data_types(args.inputfile, args.configfile)



if __name__ == "__main__":
    args = parse_cmdline()
    main(args)
