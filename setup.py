from setuptools import setup
from mwa_cal import version
import os
import json

data = [version.git_origin, version.git_hash,
        version.git_description, version.git_branch]
with open(os.path.join('mwa_cal', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths


data_files = package_files('mwa_cal', 'data')

setup_args = {
    'name':         'mwa_cal',
    'author':       'Chuneeta Nunhokee',
    'url':          'https://github.com/Chuneeta/mwa_cal',
    'license':      'BSD',
    'version':      version.version,
    'description':  'MWA Autocorrelation Calibration.',
    'packages':     ['mwa_cal'],
    'package_dir':  {'mwa_cal': 'mwa_cal'},
    'package_data': {'mwa_cal': data_files},
    'install_requires': ['numpy', 'scipy',
                         'astropy>3.0.0', 'matplotlib>=2.2',
                         'python_dateutil>=2.6.0',
                         'pytest','mwa_qa'],
    'include_package_data': True,
    'zip_safe':     False,
    'scripts': ['scripts/run_autocal.py']
}

if __name__ == '__main__':
    setup(*(), **setup_args)

