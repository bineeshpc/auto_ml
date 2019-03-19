#! /usr/bin/env python

import houseprices
import argparse
import pandas as pd
import numpy as np
import os
import utils
import config_parser

def parse_cmdline():
    parser = argparse.ArgumentParser(description='post stacker transformer')
    parser.add_argument('X_train',
                        type=str,                    
                        help='X train')

    parser.add_argument('configfile',
                        type=str,                    
                        help='configfile')
    args = parser.parse_args()
    return args

def main(args):
    df = pd.read_csv(args.X_train)
    df = houseprices.encode_features(df)
    df = houseprices.create_new_features(df)
    df = houseprices.combinations_of_features(df)
    df = houseprices.polynomial_features(df)
    for column in df.columns:
        print(column, df[column].dtype)
        try:
            if (df[column].str.contains('No')).sum() != 0:
                print(column)
        except AttributeError:
            print('exception ', column)
    print(utils.get_columns_with_null(df))
    df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
    component = 'post_stacker_transformer'
    cnf = config_parser.get_configuration(args.configfile)
    directory = cnf.get_directory(component)
    basename = os.path.basename(args.X_train)
    filename = os.path.join(directory, basename)
    df.to_csv(filename)

if __name__ == "__main__":
    args = parse_cmdline()
    main(args)
