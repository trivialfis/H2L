'''
File:          lines_data_loader.py
Author:        fis
Created:       27 Jan
Last modified: 27 Jan 2017
'''

import pickle


def segmentationData():
    linesPath = '../resource/lines.pkl'
    with open(linesPath, 'rb') as f:
        lines = pickle.load(f)
    images, labels = lines
    return images, labels
