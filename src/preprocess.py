#!/bin/python3
'''
File:          preprocess.py
Author:        fis
Created:       Feb 15 2017
Last modified: Feb 16 2017

Commentary:

'''
# from preprocessing import words_preprocess as wp
# from preprocessing import lines_preprocess as lp
# from preprocessing import words_slant_correct
# from preprocessing import alphabet_preprocess as ap
# from preprocessing import manipulators_preprocess as mp
# from multiprocessing import Pool, cpu_count
# from random import shuffle
# import numpy as np

from preprocessing import characters_preprocess as cp
from preprocessing import split_collected as sc

# def baseWords():
#     wordsList = wp.loadXMLs(wp.XML_PATH)
#     cpus = cpu_count()

#     # wordsList = wordsList[1:1000]
#     # cpus = 1

#     batch = len(wordsList)//cpus
#     tasks = [wordsList[batch*i: batch*(i+1)]
#              for i in range(cpus-1)]
#     tasks.append(wordsList[batch*(cpus-1):])
#     pool = Pool(processes=cpus)
#     imagesBufferList = pool.map(wp.start, tasks)
#     imagesBuffer = []
#     for part in imagesBufferList:
#         imagesBuffer += part
#     shuffle(imagesBuffer)
#     images, labels = zip(*imagesBuffer)
#     images = np.array(images, np.float32)
#     labels = np.array(labels, np.float32)
#     wp.save((images, labels))


# def lines():
#     lp.start()


# def slantcorrectWords():
#     wordsList = wp.loadXMLs(wp.XML_PATH)
#     cpus = cpu_count()
#     batch = len(wordsList)//cpus
#     tasks = [wordsList[batch*i: batch*(i+1)]
#              for i in range(cpus-1)]
#     pool = Pool(processes=cpus)
#     pool.map(words_slant_correct.start, tasks)


# def characters_o():
#     try:
#         # ap.start()
#         mp.start()
#     except KeyboardInterrupt:
#         print('\nExit')


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
    # slantcorrectWords()
    # lines()
    # baseWords()
    split()
    # characters()
