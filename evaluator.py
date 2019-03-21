#! /usr/bin/env python


"""
Do K fold cross validation
Select a winning model
Save it for later use
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
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
    
    parser.add_argument('y_test',
                        type=str,                    
                        help='y test file')
    
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


def evaluator(X, y, model, problem_type, configfile):
    X_test = X.values
    y_test = y.values.reshape(1, -1)[0]
    y_pred = model.predict(X_test)
    print(problem_type)
    if problem_type == 'regression':
        print('Absolute error is ', mean_absolute_error(y_test, y_pred))
        print('Mean squared error is ', mean_squared_error(y_test, y_pred))
    if problem_type == 'classification':
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
    df = pd.DataFrame({y.columns[0] : y_pred})
    prediction_file = config_parser.get_configuration(configfile).get_file_location('evaluator', 'prediction_file')
    df.to_csv(prediction_file, index=False)
    


def main(X_test, y_test, model_file, problem_type, configfile):
    evaluator_logger = logger.get_logger('evaluator',
                                         'evaluator',
                                         configfile)

    
    df1 = pd.read_csv(X_test)
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    df2 = pd.read_csv(y_test)
    evaluator(df1, df2, model, problem_type, configfile)
    
if __name__ == "__main__":
    args = parse_cmdline()
    main(args.X_test, args.y_test, args.model_file, args.problem_type, args.configfile)
