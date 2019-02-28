#! /usr/bin/env python

"""
Do K fold cross validation
Select a winning model
Save it for later use

"""

from sklearn.model_selection import train_test_split
import logger
import config_parser
import utils
import argparse
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import  Pipeline
from sklearn.linear_model import Lasso, Ridge, BayesianRidge, SGDRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import RobustScaler


model_selector_logger = logger.get_logger('model_selector',
'model_selector' )


def parse_cmdline():
    parser = argparse.ArgumentParser(description='model selector')
    parser.add_argument('inputfile',
                        type=str,                    
                        help='Inputfile')
    parser.add_argument('y',
                        type=str,                    
                        help='y column value to predict')
    
    args = parser.parse_args()
    return args


class ModelSelector:
    def __init__(self, problem_type, df, y):
        self.regressors = [
        ('linear_regression', LinearRegression()),
        ('lasso', Lasso(alpha=.5)),
        ('ridge', Ridge(alpha=.5)),
        ('bayesian_ridge', BayesianRidge()),
        ('adaboost_regressor', AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                  n_estimators=300, random_state=42)),
        ('sgd_regressor', SGDRegressor())
        ]
        self.problem_type = problem_type
        self.df = df
        self.df1 = y
        self.X = self.df.values
        self.y = self.df1.values.reshape(1, -1)[0]
        self.cv_score = dict()
        
        for regressor_name, regressor in self.regressors:
            self.create_pipeline(regressor, 'most_frequent')
            self.cv_score[regressor_name] = self.build_model()

        min_ = 10000000 # arbitrary
        min_regressor = None
        for regressor_name, values in self.cv_score.items():
            mean_ = sum(values) / len(values)
            if mean_ < min_:
                min_regressor = regressor_name
                min_ = mean_
        print(min_regressor, mean_)
        self.build_winning_model(min_regressor)
        

    def create_imputer(self, strategy):
        """ Strategy can be 'most_frequent' 'mean' etc
        """
        return 'imputer', SimpleImputer(missing_values=np.nan, strategy=strategy)

    def create_scaler(self):
        return 'scaler', RobustScaler()

    def create_regressor(self, regressor):
        return 'regressor', regressor

    def create_pipeline(self, regressor, imputation_type):
        """ Steps is a list containing an imputer, scaler and regressor
        """ 
        self.steps = [self.create_imputer(imputation_type), 
        self.create_scaler(),
        self.create_regressor(regressor)
        ]
        self.pipeline = Pipeline(self.steps)

    def build_model(self):
        return cross_val_score(self.pipeline,
         self.X,
         self.y, cv=5)

    def build_winning_model(self, regressor_name):
        for regressor_name_, regressor in self.regressors:
            if regressor_name_ == regressor_name:
                break
        self.create_pipeline(regressor, 'most_frequent')
        self.winning_model = self.pipeline.fit(self.X, self.y)
        
    def save(self):
        filename = config_parser.configuration.get_file_location('model_selector', 'model_filename')
        with open(filename, 'wb') as f:
            pickle.dump(self.winning_model, f)
        

def main(args):
    if args.inputfile:
        df = pd.read_csv(args.inputfile)
        df1 = pd.read_csv(args.y)
        model = ModelSelector('regression', df, df1)
        model.save()

if __name__ == "__main__":
    args = parse_cmdline()
    main(args)
