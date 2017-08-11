#!/usr/bin/python3
'''
File:          train.py
Author:        fis
Created:       Feb  15 2017
Last modified: July 17 2017
'''
from trainer import characterRecognizerTrainer


def trainCharacterRecognizer():
    model = characterRecognizerTrainer.trainer()
    try:
        model.train()
    except KeyboardInterrupt:
        print('\nExit')


if __name__ == '__main__':
    trainCharacterRecognizer()
