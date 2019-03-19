#! /usr/bin/env python


"""
Submit the result to kaggle
"""

import config_parser
import argparse
import os
import time

def parse_cmdline():
    parser = argparse.ArgumentParser(description='Input directory')
    parser.add_argument('resultfile',
                        type=str,                    
                        help='resultfile')
    parser.add_argument('message',
                        type=str,                    
                        help='message')
    parser.add_argument('configfile',
                        type=str,                    
                        help='Configfile')

    args = parser.parse_args()
    return args

    
def main(args):            
    cnf = config_parser.get_configuration(args.configfile)
    directory = cnf.get_directory('predictor')
    filename = args.resultfile
    filename = os.path.join(directory, filename)
    competition = cnf.configuration['competition_name']
    submit_cmd = 'kaggle competitions submit -c {competition} -f {prediction_file} -m "{message}"'.format(competition=competition,
                                                                                                       prediction_file=filename,
                                                                                                       message=args.message
    )
    print(submit_cmd)
    os.system(submit_cmd)
    check_status_cmd = 'kaggle competitions submissions -c {competition}'.format(competition=competition)
    for i in range(5):
        print(check_status_cmd)
        os.system(check_status_cmd)
        time.sleep(30)
    

if __name__ == "__main__":
    args = parse_cmdline()
    
    main(args)
