import os
import tensorflow as tf
import numpy as np
import os
import random
import cv2

class DataGenerator(tf.keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, image_paths, annot_paths, batch_size=32,
                 shuffle=True, augment=False):
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()


    def process_img(self, img):
        return img/255

    def process_mask(self, mask):
        mask = mask/255
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        return mask

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        image_paths = [self.image_paths[k] for k in indexes]
        annot_paths = [self.annot_paths[k] for k in indexes]

        X, y = self.__data_generation(image_paths, annot_paths)

        return X, y


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths, annot_paths):

        X = np.empty((self.batch_size, 512, 512, 3), dtype=np.float32)
        Y = np.empty((self.batch_size, 512, 512, 1),  dtype=np.float32)

        for i, (im_path, annot_path) in enumerate(zip(image_paths, annot_paths)):

            img = cv2.imread(im_path)
            img = self.process_img(img)
            mask = cv2.imread(annot_path, 0)
            mask = self.process_mask(mask)
            mask = mask.reshape((512,512,1))

            X[i,] = img
            Y[i,] = mask

        return X, Y
