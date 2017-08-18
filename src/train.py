#!/usr/bin/python3
'''
File:          train.py
Author:        fis
Created:       Feb  15 2017
Last modified: July 17 2017
'''
from trainer import characterRecognizerTrainer
from evaluator import h2l_debug
h2l_debug.H2L_DEBUG = True


def trainCharacterRecognizer():
    model = characterRecognizerTrainer.trainer()
    try:
        model.train()
    except KeyboardInterrupt:
        print('\nExit')


if __name__ == '__main__':
    trainCharacterRecognizer()
