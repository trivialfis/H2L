#!/bin/python3
'''
File:          preprocess.py
Author:        fis
Created:       Feb 15 2017
Last modified: Sep 11 2017

Commentary:

'''
from preprocessing import characters_preprocess as cp
from preprocessing import split_collected as sc


def characters():
    try:
        cp.start()
    except KeyboardInterrupt:
        print('\nExit')


def split():
    try:
        sc.start()
    except KeyboardInterrupt:
        print('\nExit')


if __name__ == '__main__':
    # split()
    characters()
