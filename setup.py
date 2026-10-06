"""evoparam setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('evoparam', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.rst') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name = 'evoparam',
    version = __version__,
    description = 'Parameterizing evoked neural responses',
    long_description = long_description,
    long_description_content_type = 'text/x-rst',
    python_requires = '>=3.10',
    author = 'The evoparam authors',
    url = 'https://github.com/yannbellec/evoparam',
    packages = find_packages(),
    license = 'Apache License, 2.0',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    platforms = 'any',
    project_urls = {
        'Bug Reports' : 'https://github.com/yannbellec/evoparam/issues',
        'Source' : 'https://github.com/yannbellec/evoparam'
    },
    keywords = ['neuroscience', 'evoked responses', 'ERP', 'CCEP', 'SPES', 'iEEG',
                'electrophysiology', 'parameterization'],
    install_requires = install_requires,
    tests_require = ['pytest'],
    extras_require = {
        'plot'    : ['matplotlib'],
        'data'    : ['pandas'],
        'tests'   : ['pytest'],
        'all'     : ['matplotlib', 'pandas', 'tqdm', 'pytest']
    }
)
