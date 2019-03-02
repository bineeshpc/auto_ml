#! /usr/bin/env python

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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

hyper_parameter_selector_logger = logger.get_logger('hyper_parameter_selector',
'hyper_parameter_selector' )


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
    
    def get_model(self, problem_type, df, y, model_name):
        self.problem_types = {'regression': Regressor(df, y),
                              'classification': Classifier(df, y, model_name)}
        return self.problem_types[problem_type]

class Classifier:
    
    """
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    """
    def __init__(self, df, y, model_name):
        names = ["Nearest Neighbors",
                 "Linear SVM",
                 "RBF SVM",
                 "Gaussian Process",
                 "Decision Tree",
                 "Random Forest",
                 "Neural Net",
                 "AdaBoost",
                 "Naive Bayes",
                 "QDA"
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear"),
            SVC(),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            MLPClassifier(),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()
        ]
        
        self.classifiers = dict(zip(names, classifiers))
        self.df = df
        self.df1 = y
        self.X = self.df.values
        self.y = self.df1.values.reshape(1, -1)[0]

        self.create_pipeline(self.classifiers[model_name],
                             'most_frequent'
        )
        self.select_hyper_params(model_name)
        self.build_winning_model()

        
        
    def select_hyper_params(self, model_name):
        self.hyper_params = {
            'RBF SVM': {
                'C': np.logspace(-2, 10, 2),
                'gamma': np.logspace(-9, 3, 2)
                 }
            }
        self.grid = GridSearchCV(self.classifiers[model_name],
                            param_grid=self.hyper_params[model_name],
                            cv=5)
        
        
    def create_imputer(self, strategy):
        """ Strategy can be 'most_frequent' 'mean' etc
        """
        return 'imputer', SimpleImputer(missing_values=np.nan, strategy=strategy)

    def create_scaler(self):
        return 'scaler', RobustScaler()

    def create_classifier(self, classifier):
        return 'classifier', classifier

    def create_pipeline(self, classifier, imputation_type):
        """ Steps is a list containing an imputer, scaler and regressor
        """ 
        self.steps = [self.create_imputer(imputation_type), 
        self.create_scaler(),
        self.create_classifier(classifier)
        ]
        self.pipeline = Pipeline(self.steps)



    def build_winning_model(self):
        self.winning_model = self.grid.fit(self.X, self.y)
        print("The best parameters are %s with a score of %0.2f"
      % (self.grid.best_params_, self.grid.best_score_))

        
    def save(self):
        filename = config_parser.configuration.get_file_location('hyper_parameter_selector', 'model_filename')
        hyper_parameter_selector_logger.info('winning model is saved to {}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(self.winning_model, f)
        
        
        

class Regressor:
    def __init__(self, df, y):
        self.regressors = [
        ('linear_regression', LinearRegression()),
        ('lasso', Lasso(alpha=.5)),
        ('ridge', Ridge(alpha=.5)),
        ('bayesian_ridge', BayesianRidge()),
        ('adaboost_regressor', AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                  n_estimators=300, random_state=42)),
        ('sgd_regressor', SGDRegressor())
        ]
        
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
        filename = config_parser.configuration.get_file_location('hyper_parameter_selector', 'model_filename')
        with open(filename, 'wb') as f:
            pickle.dump(self.winning_model, f)
        
            
def main(args):
    df = pd.read_csv(args.inputfile)
    df1 = pd.read_csv(args.y)
    model = ModelSelector().get_model('classification', df, df1, 'RBF SVM')
    model.save()

if __name__ == "__main__":
    args = parse_cmdline()
    main(args)
