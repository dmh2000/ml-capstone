import numpy as np
import lib.srtm as srtm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

m = srtm.read("../data/level2/N39W120.hgt")
# img = srtm.toimage(m, size=(150, 150))
# x = img_to_array(img)
# print(x.shape)
x = srtm.subdivide(m, 16)
print(x.shape)

i = 0
b = []
for a in x:
    c = np.dstack((a, a, a))
    print(c.shape)
    b.append(c)
    i = i + 1
x = np.array(b)
print(x.shape)


# for single images
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
# print(x.shape)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
y = np.ones((256))
print(y.shape)
i = 0
for xb, yb in datagen.flow(x, y, batch_size=32, save_to_dir='preview', save_prefix='img', save_format='jpeg'):
    break  # otherwise the generator would loop indefinitely
