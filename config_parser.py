import yaml
import os
import utils
import argparse

def parse_cmdline():
    parser = argparse.ArgumentParser(description='ConfigParser')
    parser.add_argument('configfile',
                        type=str,                    
                        help='configfile',
                        default='config.yml',
                        nargs='?')
    
    args = parser.parse_args()
    return args


## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return '/'.join([str(i) for i in seq])


## define custom tag handler
def join_underscore(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])


## register the tag handler
yaml.add_constructor('!join', join)
yaml.add_constructor('!join_underscore', join_underscore)


class ConfigParser:
    def __init__(self, configfile='config.yml'):
        self.configfile = configfile
        stream = open(self.configfile, 'r')
        self.configuration = yaml.load(stream)

    def get_directory(self, component):
        final_dir = self.configuration[component]['directory']
        base_dir = self.configuration['output_location']
        directory = os.path.join(base_dir, final_dir)
        utils.generate_directories(directory)
        return directory

    def get_attribute(self, component, attribute):
        return self.configuration[component][attribute]

    def get_file_location(self, component, attribute):
        directory = self.get_directory(component)
        filename = self.get_attribute(component, attribute)
        return os.path.join(directory, filename)

    def get_log_location(self, location):
        return os.path.join(self.configuration['output_location'],
        self.configuration[location]['log_location'])

    def get_components_ordered(self):
        return self.get_attribute('run_endtoend', 'order')

    def get_arguments(self, component):
        return self.configuration['run_endtoend'][component]

    def print(self):
        print(self.configuration)

    def get_source_location(self):
        return self.configuration['src_location']


def get_configuration(configfile):
    configuration = ConfigParser(configfile)
    return configuration
    

if __name__ == "__main__":

    args = parse_cmdline()
    configuration = ConfigParser(args.configfile)
    configuration.print()
