'''
File:          train.py
Author:        fis
Created:       Feb 15 2017
Last modified: Feb 16 2017
'''
from trainer import characterRecognizerTrainer
from trainer import baseSegmenterTrainer
from trainer import wordSegmenterTrainer


def trainCharacterRecognizer():
    model = characterRecognizerTrainer.trainer()
    try:
        model.train()
    except KeyboardInterrupt:
        print('\nExit')


def trainBaseCharacterSegmenter():
    model = baseSegmenterTrainer.trainer()
    try:
        model.train()
    except KeyboardInterrupt:
        print('\nExit')


def trainWordSegmenter():
    model = wordSegmenterTrainer.trainer()
    try:
        model.train()
    except KeyboardInterrupt:
        print('\nExit')


if __name__ == '__main__':
    # trainBaseCharacterSegmenter()
    trainCharacterRecognizer()
    # trainWordSegmenter()
