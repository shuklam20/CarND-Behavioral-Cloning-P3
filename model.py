# Import all the libraries
import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.utils import shuffle
import random
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2

samples = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	for line in reader:
		samples.append(line)

print("Num of Samples: " + str(len(samples)))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# some image manipulations
def brighten(image):
    # First make it bright 
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2] * random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


# Generator yields at every step, thereby preventing from the memory leak
def generator(samples, batch_size=32):
	correction = 0.4 # this is a parameter to tune
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			imgdir = '../data/IMG/'
			for batch_sample in batch_samples:
				# Center images
				filename = batch_sample[0].split('/')[-1]
				name = imgdir + filename
				image = mpimg.imread(name)#cv2.imread(name) -- changed to read in RGB
				angle = float(batch_sample[3])
				images.append(image)
				angles.append(angle)

				# left images
				filename_l = batch_sample[1].split('/')[-1]
				name_l = imgdir + filename_l
				image_l = cv2.imread(name_l)
				angle_l = float(batch_sample[3]) + correction
				images.append(image_l)
				angles.append(angle_l)

				# right images
				filename_r = batch_sample[2].split('/')[-1]
				name_r = imgdir + filename_r
				image_r = cv2.imread(name_r)
				angle_r = float(batch_sample[3]) - correction
				images.append(image_r)
				angles.append(angle_r)


			# Add more Augmented images (flipped images)
			augmented_images, augmented_angles = [], []
			angle = np.random.uniform(-30, 30) # angle for rotation
			dx, dy = np.random.randint(-3, 3, 2)
			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				augmented_images.append(cv2.flip(image,1))
				augmented_angles.append(angle*-1)

				augmented_images.append(brighten(image))
				augmented_angles.append(angle)


			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# CNN Model
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) #Top 70 pixels and bottom 25 pixels cropped
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Convolution2D(64, 3, 3,activation='relu'))
model.add(Convolution2D(64, 3, 3,activation='relu'))
model.add(Flatten())
model.add(Dense(1164, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(100, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(50, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer = l2(0.001)))
model.add(Dense(1, W_regularizer = l2(0.001)))

adam = Adam(lr = 0.0001)
model.summary()

model.compile(loss='mse', optimizer=adam)
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.fit_generator(train_generator, samples_per_epoch = len(train_samples),
	validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=40)

model.save('model.h5')
