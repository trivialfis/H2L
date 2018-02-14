from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='H2L',
    version='0.1',
    author='GamingJyun',
    author_email='gaming-jyun@outlook.com',
    license='GPL-v3',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.12.1', 'tensorflow-gpu>=1.2.1', 'tqdm',
        'opencv-python>=3.3.0', 'keras>=2.1.2', 'scikit-learn>=0.19.0',
        'pydot>=1.2.4'],
    description=("Experment project for recognizing math equations."),
    long_description=read('README.org'),
    project_urls={
        'Source Code': 'https://github.com/trivialfis/H2L'
    }
)
