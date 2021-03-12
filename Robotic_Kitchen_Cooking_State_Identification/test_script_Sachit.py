import os, cv2
import sys
import numpy as np
from keras import layers
from keras import models
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import Adam
from keras.layers import  Conv2D,MaxPooling2D,Activation,Flatten,Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,load_img
import matplotlib.pyplot as plt
import json
from keras.models import load_model


def preDataProcess(testDirectory):

	testDataGenerator = ImageDataGenerator(rescale=1./255)
	testGenerator = testDataGenerator.flow_from_directory(
	        testDirectory,
	        target_size=(128, 128),
	        batch_size=32)
	return testGenerator

#modelLoc holds path for weights file
def modelAccuracy(testGenerator):

	modelLoc = 'your path for weights.h5 file goes here'
	model = load_model(modelLoc)
	test_loss, test_accuracy = model.evaluate_generator(testGenerator)
	return test_accuracy


def main():
#testDirectory holds the path for test data
	testDirectory = 'your path for test data goes here'
	testGenerator = preDataProcess(testDirectory)
	testAccuracy = modelAccuracy(testGenerator)
#This prints the final test accuracy
	print(testAccuracy)

if __name__ == '__main__':
	main()


