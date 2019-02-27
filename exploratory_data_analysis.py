#! /usr/bin/env python
import logger
import config_parser
import argparse
import nbformat as nbf
import pandas as pd
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns

eda_logger = logger.get_logger('exploratory data analysis',
'exploratory_data_analysis' )
def parse_cmdline():
    parser = argparse.ArgumentParser(description='exploratory data analysis in python')
    parser.add_argument('inputfile',
                        type=str,                    
                        help='Inputfile')
    parser.add_argument('outputfile',
                        type=str,                    
                        help='outputfile')
    
    args = parser.parse_args()
    return args


def read_to_df(filename):
    df = pd.read_csv(filename)
    return df


def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    outliers = []
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

def detect_outlier_iqr(data_):
    pass

def find_outliers(data_1, alg='detect_outlier'):
    algs = {'detect_outlier': detect_outlier,
    'detect_outlier_iqr': detect_outlier_iqr }
    return algs[alg](data_1)


def get_df_info(df):
    sio = io.StringIO()
    df.info(buf=sio, verbose=True)
    return sio.getvalue() 

def generate_plots(type_, data, name, plots_location):
    """ Given a plot type and an data generate a plot
    """
    plt.cla()
    plt.clf()
    plot_type = getattr(sns, type_)
    plot_ = plot_type(data)
    fig = plot_.get_figure()
    fig.savefig('{}/{}_{}.png'.format(plots_location, name, type_))

def generate_plots_df(type_, df, name, plots_location):
    """ Given a plot type and an data generate a plot
    """
    plt.cla()
    plt.clf()
    if type_ == 'heatmap':
        plot_ = sns.heatmap(df, annot=True, fmt=".1f")
    if type_ == 'correlation':
        plot_ = sns.heatmap(df.corr(), annot=True, fmt=".1f")
    fig = plot_.get_figure()
    fig.savefig('{}/{}_{}.png'.format(plots_location, name, type_))


def generate_directories(directory):
    if os.path.exists(directory):
        return
    else:
        os.mkdir(directory)


def simple_analysis(df):
    cnf = config_parser.configuration
    plots_location = os.path.join(cnf['output_location'],
    cnf['exploratory_data_analysis']['plots'])
    generate_directories(plots_location)
    eda_logger.info(df.shape)
    eda_logger.info(df.head(5))
    eda_logger.info(str(df.describe()))
    eda_logger.info(get_df_info(df))
    for column in df.columns:
        eda_logger.info("Here are the outliers in column {}".format(column))
        outliers = find_outliers(df[column])
        eda_logger.info(outliers)
        fraction_of_outliers = len(outliers) / len(df[column])
        eda_logger.info("fraction of outliers is {}".format(fraction_of_outliers))
        
        unique_values = df[column].unique()
        num_unique_values = len(unique_values)
        eda_logger.info("number of unique values in column {} is {}".format(column, num_unique_values))
        value_counts = df[column].value_counts()
        eda_logger.info("Top 10 value counts for the column {} are \n{}".format(column, value_counts.head(10)))
        
        for plot_type in ['boxplot', 'violinplot']:
            generate_plots(plot_type, df[column], column, plots_location)

    generate_plots_df('heatmap', df, 'df', plots_location)
    generate_plots_df('correlation', df, 'df', plots_location)


def main(args):
    if args.inputfile:
        df = read_to_df(args.inputfile)
        simple_analysis(df)



if __name__ == "__main__":
    args = parse_cmdline()
    main(args)