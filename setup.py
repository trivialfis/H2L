from setuptools import setup, find_packages
import os
from H2L.configuration import dependencies as deps

build_dependencies = deps.build_time(deps.H2L_DEPENDENCIES)
dependencies_list = [dep[0] + '>=' + dep[1] for dep in build_dependencies]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='H2L',
    version='0.1',
    author='GamingJyun',
    author_email='gaming-jyun@outlook.com',
    license='GPL-v3',
    packages=find_packages(),
    install_requires=dependencies_list,
    tests_require=['pytest'],
    scripts=['h2l.py', 'h2l_commands.py'],
    description=("Experment project for recognizing math equations."),
    long_description=read('README.org'),
    project_urls={
        'Source Code': 'https://github.com/trivialfis/H2L'
    }
)
