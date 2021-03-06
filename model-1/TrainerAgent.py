from Constants import *

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


class TrainerAgent:

    def __init__(self, train, test=None):
        self.epochs = 100
        self.batchSize = 1

        self.train = train
        self.test = test

        self.Y_train = train["label"]  #.astype(int)
        self.X_train = train.drop(labels=["label"], axis=1)

        if MANUAL_TEST_DATA:
            self.Y_val = test["label"]
            self.X_val = test.drop (labels=["label"], axis=1)
        else:
            self.Y_val = None
            self.X_val = None

        self.model = None
        self.datagen = None
        self.history = None


    def perform(self):
        self.preprocess()
        if not MANUAL_TEST_DATA:
            self.split()
        self.constructModel()
        self.compileModel()
        self.augment()
        self.fitModel()
        self.evaluate()

    ''' ---CNN TRAINING FUNCTIONS--- '''

    def constructModel(self):
        r0 = RESOLUTION[0]
        r1 = RESOLUTION[1]

        self.model = Sequential()
        #
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                              activation='relu', input_shape=(r0, r1, NRCHANNELS)))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        #
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        # fully connected
        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(NRIMAGES, activation="softmax"))

        print(self.model.summary())

    def compileModel(self):
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    def augment(self):
        self.datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # dimension reduction
                rotation_range=90,  # randomly rotate images in the range 90 degrees
                zoom_range=0.2,  # Randomly zoom image 20%
                width_shift_range=0.2,  # randomly shift images horizontally 20%
                height_shift_range=0.2,  # randomly shift images vertically 20%
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True)  # randomly flip images

        self.datagen.fit(self.X_train)

    def fitModel(self):
        self.history = self.model.fit_generator(self.datagen.flow(self.X_train, self.Y_train, batch_size=self.batchSize),
                                                epochs=self.epochs, validation_data=(self.X_val, self.Y_val),
                                                steps_per_epoch=self.X_train.shape[0] // self.batchSize)

    def evaluate(self):
        print("Prediction:")
        print(self.model.predict(self.X_val))

        # Plot the loss and accuracy curves for training and validation
        plt.plot(self.history.history['val_loss'], color='b', label="validation loss")
        plt.title("Test Loss")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # confusion matrix
        # Predict the values from the validation dataset
        Y_pred = self.model.predict(self.X_val)
        # Convert predictions classes to one hot vectors
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        # Convert validation observations to one hot vectors
        Y_true = np.argmax(self.Y_val, axis=1)
        # compute the confusion matrix
        confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
        # plot the confusion matrix
        f, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

    def split(self):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train,
                                                                              test_size=0.1, random_state=2)

    ''' ---DATA PREPROCESSING FUNCTIONS--- '''

    def preprocess(self):
        self.normalize()
        self.reshape()
        self.encode()

    def encode(self):
        self.Y_train = to_categorical(self.Y_train)
        self.Y_val   = to_categorical(self.Y_val)

    def reshape(self):
        r0 = RESOLUTION[0]
        r1 = RESOLUTION[1]
        self.X_train = self.X_train.values.reshape(-1, r0, r1, NRCHANNELS)
        self.X_val   = self.X_val.values.reshape(-1, r0, r1, NRCHANNELS)

    def normalize(self):
        self.X_train = self.X_train / 255.0
        self.X_val   = self.X_val / 255.0
