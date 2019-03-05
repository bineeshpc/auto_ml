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

args = parse_cmdline()
configuration = ConfigParser(args.configfile)
# def generate_files():
#     import os
#     content = "#! /usr/bin/env python\n"
#     for k, v in y.items():
#         with open(v, "w") as f:
#             f.write(content)
#         os.system('chmod u+x {}'.format(v))
        
# generate_files()
if __name__ == "__main__":
    
    configuration.print()
