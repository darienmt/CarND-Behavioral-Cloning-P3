import cv2
import csv
import numpy as np

def getLinesFromDrivingLogs(dataPath, skipHeader=False):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def loadImagesAndMeasurements(dataPath, skipHeader=False):
    """
    Loads the images and measurements from the driving logs in the directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    Returns a pair `(images, measurements)`
    """
    lines = getLinesFromDrivingLogs(dataPath, skipHeader)
    images = []
    measurements = []
    for line in lines:
        imagePath = line[0]
        image = cv2.cvtColor(cv2.imread(dataPath + '/' + imagePath), cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        # Flipping
        images.append(cv2.flip(image,1))
        measurements.append(measurement*-1.0)

    return (np.array(images), np.array(measurements))

X_train, y_train = loadImagesAndMeasurements('data/data', skipHeader=True)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers.pooling import MaxPooling2D

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    return model

def leNetModel():
    """
    Creates a LeNet model.
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def trainAndSave(model, inputs, outputs, modelFile, epochs = 3):
    """
    Train the model `model` using 'mse' lost and 'adam' optimizer for the epochs `epochs`.
    The model is saved at `modelFile`
    """
    model.compile(loss='mse', optimizer='adam')
    model.fit(inputs, outputs, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save(modelFile)
    print("Model saved at " + modelFile)

model = leNetModel()
trainAndSave(model, X_train, y_train, 'models/data.h5')
