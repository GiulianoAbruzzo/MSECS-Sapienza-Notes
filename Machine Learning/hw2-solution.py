import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras import models, layers, backend
from keras.layers import Dense, Activation, Flatten, Conv2D, AveragePooling2D
from keras.models import Sequential
import shutil
from functools import reduce
import os
import re

def CNN_model(shape, classes_number):
	cnn_model = Sequential()
	cnn_model.add(Conv2D(6, kernel_size=(3, 3), strides=(1, 1), activation='tanh', input_shape=shape, padding="same"))
	cnn_model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	cnn_model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='valid'))
	cnn_model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
	cnn_model.add(Conv2D(120, kernel_size=(3, 3), strides=(1, 1), activation='tanh', padding='valid'))
	cnn_model.add(Flatten())
	cnn_model.add(Dense(84, activation='tanh'))
	cnn_model.add(Dense(classes_number, activation='softmax'))
	cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
	return cnn_model

def results(path, num_classes):
	labels_name = next(os.walk(path))[1]
	filename_list = list()
	label_list = list()
	classes = dict()
	class_id = [0] * num_classes
	for class_name in labels_name:
		index = labels_name.index(class_name)
		current_id = class_id.copy()
		current_id[index] += 1
		classes[class_name] = current_id
		img_list = os.listdir(path + "/" + class_name)
		for elem in img_list:
			if (elem.startswith('.')):
		        	img_list.remove(elem)
		for img in img_list:
			filename_list.append(path + "/" + class_name + "/" + img)
			label_list.append(index)
	return filename_list, label_list, classes

def create_training_set(boats, num_classes):
	if not os.path.exists('./reducedTraining'):
        	os.makedirs('./reducedTraining')
	else:
		shutil.rmtree('./reducedTraining')
		os.makedirs('./reducedTraining')
	for cartella in os.listdir('./sc5'):
		if '.txt' not in cartella:
			if cartella in boats:
				for files in os.listdir('./sc5/'+cartella):
					if not os.path.exists('./reducedTraining/'+cartella):
						os.makedirs('./reducedTraining/'+cartella)
					shutil.copy2(os.path.join('./sc5/'+cartella,files),os.path.join('./reducedTraining/'+cartella,files))
	return results('./reducedTraining',num_classes)

def create_test_set(boats, num_classes):
	if not os.path.exists('./reducedTest'):
		os.makedirs('./reducedTest')
	else:
		shutil.rmtree('./reducedTest')
		os.makedirs('./reducedTest')

	class_list = open("./sc5-2013-Mar-Apr-Test-20130412/ground_truth.txt", "r")
	for line in class_list:
		temp = line.split(";")
		filename = temp[0]
		category = temp[1]
		word_list = re.split('[^a-zA-Z0-9]', category)
		if not "Snapshot" in word_list:
			boat_class = reduce((lambda x, y: x + y), word_list)
			if not os.path.exists('./reducedTest/'+boat_class):
				os.makedirs('./reducedTest/'+boat_class)
			shutil.copy2(os.path.join('./sc5-2013-Mar-Apr-Test-20130412/',filename), os.path.join('./reducedTest/'+boat_class,filename))
	class_list.close()

	for cartella in os.listdir('./reducedTest'):
		if not cartella in boats:
			shutil.rmtree('./reducedTest/'+cartella, ignore_errors=True)    
	return results('./reducedTest',num_classes)

target_classes=['Alilaguna','Ambulanza','Barchino','Lanciafino10mBianca','Lanciafino10mMarrone','Motobarca','Mototopo','Patanella','Raccoltarifiuti','VaporettoACTV']
#target_classes=['Barchino','Lanciafino10mBianca','Motobarca','Mototopo','Raccoltarifiuti']
#target_classes=['Alilaguna','Ambulanza','Lanciafino10mMarrone','Patanella','VaporettoACTV']
classes_num = len(target_classes)

train_input = []
train_output = []
train_filename_list, train_label_list, train_classes = create_training_set(target_classes, classes_num)
train_size = len(train_filename_list)
for i in range(train_size):
	img = cv2.imread(train_filename_list[i], cv2.IMREAD_COLOR)
	label = train_label_list[i]
	img_resize = cv2.resize(img,(80,24))
	train_input.append(img_resize)
	train_output.append(label)

train_input = np.array(train_input)
train_output = np.array(train_output)
train_output = keras.utils.to_categorical(train_output, classes_num)

test_input = []
test_output = []
test_filename_list, test_label_list, test_classes = create_test_set(target_classes, classes_num)
test_size = len(test_filename_list)
for i in range(test_size):
	img = cv2.imread(test_filename_list[i], cv2.IMREAD_COLOR)
	lab = test_label_list[i]
	img_resize = cv2.resize(img,(80,24))
	test_input.append(img_resize)
	test_output.append(lab)

test_input = np.array(test_input)
test_output = np.array(test_output)
test_output = keras.utils.to_categorical(test_output, classes_num)
input_shape = (train_input.shape[1], train_input.shape[2], 3)

print("Training input %s" %str(train_input.shape))
print("Training output %s" %str(train_output.shape))
print("Test input %s" %str(test_input.shape))
print("Test output %s" %str(test_output.shape))
print("Number of classes: %d" %classes_num)

# Create model
model = CNN_model(input_shape, classes_num)
print("Training")

# Train
hist = model.fit(train_input, train_output, batch_size=32, epochs=10)
performances = model.evaluate(test_input,test_output)
print("Test Accuracy: "+str("%.2f"%performances[1]))
print("Test Loss: "+str("%.2f"%performances[0]))

# Test
Y_pred = model.predict(test_input, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)
y_test = np.argmax(test_output,axis=1)
print(confusion_matrix(y_test,y_pred))
TestSetPath = './reducedTest'
test_classes = os.listdir(TestSetPath)
print(classification_report(y_test,y_pred,target_names=test_classes))
