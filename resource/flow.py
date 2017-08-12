from keras.preprocessing import image as Image
from matplotlib import pyplot as plt

train_datagen = Image.ImageDataGenerator(
    zca_epsilon=None,
)
train_flow = train_datagen.flow_from_directory(
    './characters',
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=128
)
mapping = train_flow.class_indices
mapping = dict((v, k) for k, v in mapping.items())
with open('characters_map', 'w') as f:
    f.write(str(mapping))

validation_gen = Image.ImageDataGenerator(
    zca_epsilon=None,
)
validation_flow = validation_gen.flow_from_directory(
    './validation',
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=512
)

count = 0
for i in train_flow:
    print(type(i), len(i))
    images, labels = i
    print(len(images), len(labels))
    img = images[0]
    print(type(img), img.shape, img.dtype)
    img = img.reshape(48, 48)
    plt.imshow(img, cmap='gray')
    plt.show()
    if count == 5:
        break
    count += 1

for i in validation_flow:
    print(type(i), len(i))
    images, labels = i
    print(len(images), len(labels))
    img = images[0]
    print(type(img), img.shape, img.dtype)
    img = img.reshape(48, 48)
    plt.imshow(img, cmap='gray')
    plt.show()
    if count == 10:
        break
    count += 1
