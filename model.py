import cv2
import csv
import pandas as pd
import numpy as np

lines = []
with open('./driving_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


images = []
measurements = []
for line in lines:
	for i in range(3):
		orig_path = line[i]
		filename = orig_path.split('/')[-1]
		local_path = './driving_data/IMG/' + filename
		image = cv2.imread(local_path)
		images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement+0.2)
	measurements.append(measurement-0.2)


x_train = np.asarray(images)
y_train = np.asarray(measurements)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, ELU, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=3)

model.save('model.h5')
