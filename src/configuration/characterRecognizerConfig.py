import os

ARCHITECTURE_FILE = './models/character_recognizer_architure.json'
WEIGHTS_FILE = './models/character_recognizer_weights.hdf5'
NAME = 'character_recognizer'
CHARACTER_MAP = './models/characters_map'

BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 300

CLASS_NUM = 36

INIT_LEARNING_RATE = 2
EPOCH = 30
SAMPLES_PER_EPOCH = 648000
VALIDATION_SAMPLES = 72000
VALIDATION_STEPS = VALIDATION_SAMPLES / VALIDATION_BATCH_SIZE

IMG_ROWS, IMG_COLS = 48, 48
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

# Data loader
TRAIN_DATA = '../resource/training'
VALIDATION_DATA = '../resource/validation'
FILES_COUNT = 12


def modelExists():
    weightsExists = os.path.exists(WEIGHTS_FILE)
    architectureExists = os.path.exists(ARCHITECTURE_FILE)
    return weightsExists and architectureExists
