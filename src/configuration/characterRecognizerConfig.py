import os

ARCHITECTURE_FILE = './models/character_recognizer_architure.json'
WEIGHTS_FILE = './models/character_recognizer_weights.hdf5'
VISUAL_FILE = './models/model_plot.png'
NAME = 'character_recognizer'
CHARACTER_MAP = './models/characters_map'

BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 300

CLASS_NUM = 36

INIT_LEARNING_RATE = 2
EPOCH = 30

IMG_ROWS, IMG_COLS = 48, 48
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

# Data directories
TRAIN_DATA = '../resource/training'
VALIDATION_DATA = '../resource/validation'


def modelExists():
    weightsExists = os.path.exists(WEIGHTS_FILE)
    architectureExists = os.path.exists(ARCHITECTURE_FILE)
    return weightsExists and architectureExists
