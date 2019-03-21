#! /usr/bin/env python


"""
Do K fold cross validation
Select a winning model
Save it for later use
"""

import logger
import config_parser
import utils
import argparse
import pandas as pd
import numpy as np
import os
import pickle


def parse_cmdline():
    parser = argparse.ArgumentParser(description='model selector')
    parser.add_argument('X_test',
                        type=str,                    
                        help='test file to train')

    parser.add_argument('id_',
                        type=str,                    
                        help='id of the problem set for creating final output')

    parser.add_argument('y_column',
                        type=str,                    
                        help='y column')

    
    parser.add_argument('model_file',
                        type=str,                    
                        help='model_file')

    parser.add_argument('problem_type',
                        type=str,                    
                        help='problem type')
    
    parser.add_argument('configfile',
                        type=str,                    
                        help='configfile')

    args = parser.parse_args()
    return args

def predictor(args):
    #def predictor(X, y_column, model, problem_type, configfile):
    df1 = pd.read_csv(args.X_test)
    # print(df1.isnull().any())
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)
    X_test = df1.values
    y_pred = model.predict(X_test)
    cnf = config_parser.get_configuration(args.configfile)

    id_ = args.id_
    y_column = args.y_column
    df = pd.DataFrame({id_: range(1, 28000+1), y_column : y_pred})
    prediction_file = cnf.get_file_location('predictor', 'prediction_file')
    df.to_csv(prediction_file, index=False)
    

def main(args):
    #def main(X_test, y_column, model_file, problem_type, configfile):
    predictor_logger = logger.get_logger('predictor',
                                         'predictor',
                                         args.configfile)
    # predictor(df1, y_column, model, problem_type, configfile)
    predictor(args)
    
if __name__ == "__main__":
    args = parse_cmdline()
    #main(args.X_test, args.y_column, args.model_file, args.problem_type, args.configfile)
    main(args)
