import logging
import logging.handlers
import config_parser
import os

def generate_directories(filename):
    if os.path.exists(filename):
        return
    else:
        directories = filename.split('/')[0:-1]
        location = '/'.join(directories)
        try:
            os.mkdir(location)
        except:
            pass


def get_logger(name, location, configfile):
    configuration = config_parser.get_configuration(configfile)
    datageneratorlogfile = configuration.get_log_location(location)
    generate_directories(datageneratorlogfile)
    # create logger with 'insight data generation'
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(datageneratorlogfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    #create a log rotation handler
    rh = logging.handlers.RotatingFileHandler(datageneratorlogfile,
                maxBytes=5 * 1024 * 1024, backupCount=5)
    rh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.addHandler(rh)
    return logger
