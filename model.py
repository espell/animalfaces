import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import keras
import random
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, Input,InputLayer, Activation, BatchNormalization
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D, SpatialDropout2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import random
import cv2
import os
import numpy as np
from keras.applications.resnet50 import ResNet50
# import efficientnet.keras as efn
from keras.models import Model
import keras
from keras.callbacks import EarlyStopping

img_size = 224
num_epochs = 30
labels = ['cat', 'dog', 'wild']

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img))
                resized_arr = cv2.resize(img, (img_size, img_size))
                data.append([resized_arr, class_num]) #normal resized image
                # gray=cv2.cvtColor(resized_arr, cv2.COLOR_BGR2GRAY)
                # data.append([gray, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

print("getting train")
train_path = r'C:\Users\emmas\Documents\cs230\faces\train'
test_path = r'C:\Users\emmas\Documents\cs230\faces\val'
train=get_data(train_path)
print("getting test")
test=get_data(test_path)

x_train = []
y_train = []
x_test = []
y_test = []

for feature, label in train:
  x_train.append(feature)
  #print(label)
  y_train.append(label)

for feature, label in test:
  x_test.append(feature)
  y_test.append(label)

x_train = np.array(x_train,dtype=np.float16) / 255
x_test = np.array(x_test,dtype=np.float16) / 255

x_train=x_train.reshape(-1, img_size, img_size, 3)
y_train = np.array(y_train)

x_test=x_test.reshape(-1, img_size, img_size, 3)
y_test = np.array(y_test)

print("######################    Training and test shapes  ##########################")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print("")

model = Sequential()
model.add(InputLayer(input_shape=(img_size, img_size, 3)))

model.add(ZeroPadding2D((3, 3)))
model.add(Conv2D(112, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D((2, 2), strides=(3, 3)))
model.add(Conv2D(72, activation='relu', kernel_size=(2, 2)))
model.add(MaxPooling2D((2, 2), strides=(3, 3)))
model.add(Conv2D(64, activation='relu', kernel_size=(2, 2)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Conv2D(32, activation='relu', kernel_size=(2, 2)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(3, 3)))
model.add(Conv2D(16, activation='relu', kernel_size=(2, 2)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dropout(0.5))
model.add(Dense(128,activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()

rms= RMSprop(lr=0.00001)
model.compile(optimizer =rms , loss =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

X_train,x_valid,Y_train,y_valid = train_test_split(x_train,y_train,train_size = 0.8,test_size = 0.2,random_state =42)

history = model.fit(X_train,Y_train,epochs = 30,validation_data = (x_valid,y_valid))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(num_epochs)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_test, predictions, target_names = ['cat (Class 0)','dog (Class 1)','wild (Class 2)']))
