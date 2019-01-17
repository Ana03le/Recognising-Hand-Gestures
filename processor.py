##This part of the code is reproduced from https://github.com/harvitronix/five-video-classification-methods, authored by Matt Harvey##

from keras.preprocessing.image import img_to_array, load_img
import numpy as np

def process_image(image, target_shape):
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x
