from keras.preprocessing import image as Image
from matplotlib import pyplot as plt

train_datagen = Image.ImageDataGenerator(
    zca_epsilon=None,
)
train_generator = train_datagen.flow_from_directory(
    './characters',
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=128
)
mapping = train_generator.class_indices
mapping = dict((v, k) for k, v in mapping.items())
print(mapping)
for i in train_generator:
    print(type(i), len(i))
    images, labels = i
    print(len(images), len(labels))
    img = images[0]
    print(type(img), img.shape, img.dtype)
    img = img.reshape(48, 48)
    plt.imshow(img, cmap='gray')
    plt.show()
