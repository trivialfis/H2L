#!/usr/bin/env python3

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
