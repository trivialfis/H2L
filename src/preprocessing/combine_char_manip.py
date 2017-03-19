'''
File:          combine_char_manip.py
Author:        fis
Created date:  Feb 6 2017
Last modified: Feb 7 2017
'''
import pickle
from random import shuffle

CHARACTERS_FILE = '../resource/alphabet'
MANIPULATORS_FILE = '../resource/manipulators.pkl'
TARGET_FILE = '../resource/characters'
VALIDATION_FILE = '../resource/validation.pkl'
VALIDATION_SIZE = 800
FILE_COUNT = 12

with open(MANIPULATORS_FILE, 'rb') as f:
    manipulators = pickle.load(f)
shuffle(manipulators)
length = len(manipulators)

validationData = []
for i in range(FILE_COUNT):
    filename = CHARACTERS_FILE + str(i) + '.pkl'
    with open(filename, 'rb') as f:
        characters = pickle.load(f)
    characters += manipulators[length//FILE_COUNT*i:length//FILE_COUNT*(i+1)]
    shuffle(characters)
    shuffle(characters)
    validationData += characters[-VALIDATION_SIZE:]
    characters = characters[:-VALIDATION_SIZE]
    targetName = TARGET_FILE + str(i) + '.pkl'
    with open(targetName, 'wb') as f:
        pickle.dump(characters, f)

with open(VALIDATION_FILE, 'wb') as f:
    pickle.dump(validationData, f)
