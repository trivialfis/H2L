import os

ARCHITECTURE_FILE = './models/character_recognizer_architure.json'
WEIGHTS_FILE = './models/character_recognizer_weights.hdf5'
NAME = 'character_recognizer'
CHARACTER_MAP = '../resource/character_map'

BATCH_SIZE = 128
CLASS_NUM = 47
INIT_LEARNING_RATE = 2
EPOCH = 600
SAMPLES_PER_EPOCH = 470190
IMG_ROWS, IMG_COLS = 48, 48
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

# Data loader
TRAIN_DATA = '../resource/characters_pkl/characters'
VALIDATION_DATA = '../resource/characters_pkl/validation.pkl'
FILES_COUNT = 10


def modelExists():
    weightsExists = os.path.exists(WEIGHTS_FILE)
    architectureExists = os.path.exists(ARCHITECTURE_FILE)
    return weightsExists and architectureExists
