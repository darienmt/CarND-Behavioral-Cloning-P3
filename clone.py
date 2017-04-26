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

def loadImageAndMeasurement(dataPath, imagePath, measurement, images, measurements):
    """
    Executes the following steps:
      - Loads the image from `dataPath` and `imagPath`.
      - Converts the image from BGR to RGB.
      - Adds the image and `measurement` to `images` and `measurements`.
      - Flips the image vertically.
      - Inverts the sign of the `measurement`.
      - Adds the flipped image and inverted `measurement` to `images` and `measurements`.
    """
    originalImage = cv2.imread(dataPath + '/' + imagePath.strip())
    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurements.append(measurement)
    # Flipping
    images.append(cv2.flip(image,1))
    measurements.append(measurement*-1.0)

def loadImagesAndMeasurements(dataPath, skipHeader=False, correction=0.2):
    """
    Loads the images and measurements from the driving logs in the directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    `correction` is the value to add/substract to the measurement to use side cameras.
    Returns a pair `(images, measurements)`
    """
    lines = getLinesFromDrivingLogs(dataPath, skipHeader)
    images = []
    measurements = []
    for line in lines:
        measurement = float(line[3])
        # Center
        loadImageAndMeasurement(dataPath, line[0], measurement, images, measurements)
        # Left
        loadImageAndMeasurement(dataPath, line[1], measurement + correction, images, measurements)
        # Right
        loadImageAndMeasurement(dataPath, line[2], measurement - correction, images, measurements)

    return (np.array(images), np.array(measurements))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def trainAndSave(model, inputs, outputs, modelFile, epochs = 3):
    """
    Train the model `model` using 'mse' lost and 'adam' optimizer for the epochs `epochs`.
    The model is saved at `modelFile`
    """
    model.compile(loss='mse', optimizer='adam')
    model.fit(inputs, outputs, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save(modelFile)
    print("Model saved at " + modelFile)

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def leNetModel():
    """
    Creates a LeNet model.
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


print('Loading images')
X_train, y_train = loadImagesAndMeasurements('data/data', skipHeader=True)
# model = leNetModel()
model = nVidiaModel()
print('Training model')
trainAndSave(model, X_train, y_train, 'models/nVidea_data.h5')
print('The End')
