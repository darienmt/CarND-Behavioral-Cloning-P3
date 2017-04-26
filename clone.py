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
        image = cv2.imread(dataPath + '/' + imagePath)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    return (np.array(images), np.array(measurements))

X_train, y_train = loadImagesAndMeasurements('data/data', skipHeader=True)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('models/data.h5')
