#! /usr/bin/env python
from collections import Counter
import logger
import config_parser
import argparse
import utils
import pandas as pd
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import Image, display
import nbformat as nbf


def parse_cmdline():
    parser = argparse.ArgumentParser(description='exploratory data analysis in python')
    parser.add_argument('inputfile',
                        type=str,                    
                        help='Inputfile')
    parser.add_argument('target_column',
                        type=str,                    
                        help='target_column')
    parser.add_argument('configfile',
                         type=str,                    
                         help='configfile')
    
    args = parser.parse_args()
    return args


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


def detect_outliers(df, features, n=0):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    df  dataframe
    features iterable
    n integer
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers

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

def get_in_between(df, target_column, a, b):
    tc = target_column
    eda_logger.info("In between {} and {}".format(a, b))
    return df[(df[tc] > a) & (df[tc] <= b)][target_column]
    

def create_notebook(plots_location, notebook):
    cells = notebook['cells']

    cells.append(nbf.v4.new_code_cell("from IPython.core.display import Image, display"))
    for filename in os.listdir(plots_location):
        if 'png' in filename:
            full_filename = os.path.join(plots_location, filename)
            cells.append(nbf.v4.new_markdown_cell(full_filename))
            code_ = "display(Image(filename='{}'))".format(full_filename)
            cells.append(nbf.v4.new_code_cell(code_))

def simple_analysis(inputfile, configfile, target_column):
    df = pd.read_csv(inputfile)
    cnf = config_parser.get_configuration(configfile)
    plots_location = cnf.get_directory('exploratory_data_analysis')
    src_location = cnf.get_source_location()
    notebook = nbf.v4.new_notebook()
    cells = notebook['cells']
    src_filename = 'exploratory_data_analysis.py'
    tempfilename = '{}/tmp_{}'.format(src_location, src_filename)
    os.system('head -n -11 {} > {}'.format(src_filename, tempfilename))
    cells.append(nbf.v4.new_code_cell("%load {}".format(tempfilename)))
    notebook_start = """from collections import Counter
import logger
import config_parser
import argparse
import utils
import pandas as pd
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import Image, display
import nbformat as nbf
df = pd.read_csv('{inputfile}')
""".format(inputfile=inputfile)
    cells.append(nbf.v4.new_code_cell(notebook_start))

    eda_logger.info(df.shape)
    cells.append(nbf.v4.new_code_cell("df.shape"))
    eda_logger.info(df.head(5))
    cells.append(nbf.v4.new_code_cell("df.head(5)"))
    eda_logger.info(str(df.describe()))
    cells.append(nbf.v4.new_code_cell("df.describe()"))
    eda_logger.info(get_df_info(df))
    cells.append(nbf.v4.new_code_cell("df.info()"))
    numerical_columns = utils.get_numerical_columns(df)
    code_ = """numerical_columns = utils.get_numerical_columns(df)
numerical_columns"""
    str_ = "Numerical columns are {}".format(numerical_columns)
    eda_logger.info(str_)
    cells.append(nbf.v4.new_markdown_cell(str_))
    cells.append(nbf.v4.new_code_cell(code_))
    outliers_info = {'column': [], 'outliers_rate': []}
    for column in df.columns:
        code_ = 'column="{}"'.format(column)
        cells.append(nbf.v4.new_code_cell(code_))
        try:
            if column in numerical_columns:
                str_ = "Here are the outliers in column {}".format(column)
                eda_logger.info(str_)
                cells.append(nbf.v4.new_markdown_cell(str_))
                outliers = find_outliers(df[column])
                code_ = "outliers = find_outliers(df[column])"
                eda_logger.info(outliers)
                cells.append(nbf.v4.new_code_cell(code_))
                fraction_of_outliers = len(outliers) / len(df[column])
                code_ = "fraction_of_outliers = len(outliers) / len(df[column])"
                cells.append(nbf.v4.new_code_cell(code_))
                str_ = "fraction of outliers is {}".format(fraction_of_outliers)
                outliers_info['column'].append(column)
                outliers_info['outliers_rate'].append(fraction_of_outliers)
                eda_logger.info(str_)
                cells.append(nbf.v4.new_markdown_cell(str_))
        except:
            eda_logger.error('Exception happened while processing column {} outliers'.format(column))
        try:
            unique_values = df[column].unique()
            num_unique_values = len(unique_values)
            code_ = """unique_values = df[column].unique()
num_unique_values = len(unique_values)"""
            str_ = "number of unique values in column {} is {}".format(column, num_unique_values)
            eda_logger.info(str_)
            cells.append(nbf.v4.new_code_cell(code_))
            cells.append(nbf.v4.new_markdown_cell(str_))


            value_counts = df[column].value_counts()
            code_ = """value_counts = df[column].value_counts()
value_counts"""
            str_ = "Top 10 value counts for the column {} are \n{}".format(column, value_counts.head(10))
            eda_logger.info(str_)
            cells.append(nbf.v4.new_code_cell(code_))
            cells.append(nbf.v4.new_markdown_cell(str_))


            for plot_type in ['boxplot', 'violinplot']:
                generate_plots(plot_type, df[column], column, plots_location)
        except:
            eda_logger.error('Exception happened while processing column {}'.format(column))

    try:
        numerical_df = df[numerical_columns]
        df_corr = numerical_df.corr()
        code_ = """numerical_df = df[numerical_columns]
df_corr = numerical_df.corr()
df_corr"""
        cells.append(nbf.v4.new_code_cell(code_))
        eda_logger.info(df_corr[target_column].sort_values())
        code_ = """target_column='{}'
df_corr[target_column].sort_values(ascending=False)""".format(target_column)
        cells.append(nbf.v4.new_code_cell(code_))
        eda_logger.info(get_in_between(df_corr, target_column, .5, 1))
        eda_logger.info(get_in_between(df_corr, target_column, .0, .5))
        eda_logger.info(get_in_between(df_corr, target_column, -.5, 0))
        eda_logger.info(get_in_between(df_corr, target_column, -1, -.5))
        generate_plots_df('heatmap', numerical_df, 'numerical', plots_location)
        generate_plots_df('correlation', numerical_df, 'numerical', plots_location)

        code_ = """null_rate_numerical_df = utils.get_null_rate(numerical_df)
null_rate_numerical_df"""
        null_rate = utils.get_null_rate(numerical_df)
        eda_logger.info(null_rate)
        cells.append(nbf.v4.new_code_cell(code_))


        code_ = """null_rate_df = utils.get_null_rate(df)
null_rate_df"""
        null_rate = utils.get_null_rate(df)
        eda_logger.info(null_rate)
        cells.append(nbf.v4.new_code_cell(code_))

        code_ = """outliers_info={}
outliers_df = pd.DataFrame(outliers_info).sort_values(['outliers_rate'], ascending=False)
outliers_df""".format(outliers_info)
        outliers_df = pd.DataFrame(outliers_info).sort_values(['outliers_rate'], ascending=False)
        eda_logger.info(outliers_df)
        cells.append(nbf.v4.new_code_cell(code_))

        code_ = """negative_numbers_ratio = utils.get_negative_numbers_ratio(numerical_df)
negative_numbers_ratio"""
        negative_numbers_ratio = utils.get_negative_numbers_ratio(numerical_df)
        eda_logger.info(negative_numbers_ratio)
        cells.append(nbf.v4.new_code_cell(code_))
    except Exception as e:
        eda_logger.error('Exception happened while processing heatmap')
        print(e)
        

    # create_notebook(plots_location, notebook)
    nbf.write(notebook, os.path.join(src_location, 'eda.ipynb'))
        

def main(args):
    simple_analysis(args.inputfile, args.configfile, args.target_column)








if __name__ == "__main__":
    args = parse_cmdline()
    eda_logger = logger.get_logger('exploratory data analysis',
                                   'exploratory_data_analysis',
                                   args.configfile)
    main(args)