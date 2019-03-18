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

    original_test_file = cnf.get_attribute('run_endtoend', 'generate_datatypes')['X_train']
    # print(args.X_test)
    # print(original_test_file)
    df2 = pd.read_csv(original_test_file)
    id_ = 'Id'
    # print(set(df2.Id) - set(df1.Id))
    # # print(df2.iloc[1458])
    # print(df1[df1.Id == 2550])
    # print(df2[df2.Id == 2550])
    y_column = args.y_column
    # print(df1.shape)
    # print(df2.shape)
    df = pd.DataFrame({id_:df1[id_], y_column : np.expm1(y_pred)})
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