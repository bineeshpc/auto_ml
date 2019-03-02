#! /usr/bin/env python


"""
Do the entire sequence of process one after the other.

"""

from jobrunner import Command
import logger
import argparse
import config_parser
import os

run_endtoend_logger = logger.get_logger('run_endtoend',
'run_endtoend' )


def parse_cmdline():
    parser = argparse.ArgumentParser(description='run end to end')
    parser.add_argument('configfile',
                        type=str,                    
                        help='configuration file')
    
    args = parser.parse_args()
    return args

        

def main(args):
    """
    ./splitter.py winequality-red.csv quality


./evaluater.py /tmp/competition/splitter/X_test.csv /tmp/competition/splitter/y_test.csv /tmp/competition/model_selector/winning_model.pkl 

  ./model_selector.py /tmp/competition/splitter/X_train.csv /tmp/competition/splitter/y_train.csv


./generate_datatypes.py /tmp/competition/splitter/X_train.csv

./exploratory_data_analysis.py winequality-red.csv
    """
    commands = [
        ]
    def get_arguments(component):
        arg_dict = config_parser.configuration.get_arguments(component)
        values = []
        if 'order' in arg_dict:
            for arg_ in arg_dict['order']:
                values.append(arg_dict[arg_])
        locations = []
        for location in arg_dict['location']:
            locations.append(location)
        outs = []
        for value, location in zip(values, locations):
            if location == 'None':
                outs.append(value)
            else:
                directory = config_parser.configuration.get_directory(location)
                val = os.path.join(directory, value)
                outs.append(val)
                
        return ' '.join(outs)

    def should_run(component):
        
        arg_dict = config_parser.configuration.get_arguments(component)
        return arg_dict['run']
        

    for component in config_parser.configuration.get_components_ordered():
        src_location = config_parser.configuration.get_source_location()
        program = config_parser.configuration.get_attribute(component, 'program')
        arguments = get_arguments(component)
        command_string = '{}/{} {}'.format(src_location, program, arguments)
        cmd = Command(command_string)
        if should_run(component):
            cmd.do()
        
    
if __name__ == "__main__":
    args = parse_cmdline()
    main(args)

