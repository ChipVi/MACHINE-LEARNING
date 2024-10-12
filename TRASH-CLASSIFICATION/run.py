import tensorflow as tf
import tensorflow.python.keras.callbacks
from tensorflow.python.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
# import tensorflow.python.keras as keras
from PIL import Image
from pathlib import Path
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
# import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as T  

data_dir = Path('original_images')

transformer = T.Compose([T.Resize((32, 32)), T.ToTensor()])
dataset = ImageFolder(data_dir, transform = transformer)

# display class names
# print(dataset.classes) 

PATH_TEST = r"original_images"
PATH_TRAIN = r"after"
class_names = ['cardboard', 'glass', 'metal','paper','plastic','trash']

imagepath_cardboard = r"original_images\cardboard"
graypath_cardboard = r"after\cardboard"
File_listing = os.listdir(imagepath_cardboard)
for file in File_listing:
    im = Image.open(imagepath_cardboard + '\\' + file) 
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_cardboard + '\\' + file, "JPEG")

imagepath_glass = r"original_images\glass"
graypath_glass = r"after\glass"
File_listing = os.listdir(imagepath_glass)
for file in File_listing:
    im = Image.open(imagepath_glass + '\\' + file) 
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_glass + '\\' + file, "JPEG")

imagepath_trash = r"original_images\trash"
graypath_trash = r"after\trash"
File_listing = os.listdir(imagepath_trash)
for file in File_listing:
    im = Image.open(imagepath_trash + '\\' + file) 
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_trash + '\\' + file, "JPEG")

imagepath_plastic = r"original_images\plastic"
graypath_plastic = r"after\plastic"
File_listing = os.listdir(imagepath_plastic)
for file in File_listing:
    im = Image.open(imagepath_plastic + '\\' + file) 
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_plastic + '\\' + file, "JPEG")

imagepath_paper = r"original_images\paper"
graypath_paper = r"after\paper"
File_listing = os.listdir(imagepath_paper)
for file in File_listing:
    im = Image.open(imagepath_paper + '\\' + file) 
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_paper + '\\' + file, "JPEG")

imagepath_metal = r"original_images\metal"
graypath_metal = r"after\metal"
File_listing = os.listdir(imagepath_metal)
for file in File_listing:
    im = Image.open(imagepath_metal + '\\' + file) 
    img = im.resize((32,32))
    gray = img.convert('L')
    gray.save(graypath_metal + '\\' + file, "JPEG")

train_dir = os.path.join(PATH_TRAIN)
test_dir = os.path.join(PATH_TEST)

imagepath_cardboard_dir = os.path.join(imagepath_cardboard)
imagepath_glass_dir = os.path.join(imagepath_glass)
imagepath_metal_dir = os.path.join(imagepath_metal)
imagepath_paper_dir = os.path.join(imagepath_paper)
imagepath_plastic_dir = os.path.join(imagepath_plastic)
imagepath_trash_dir = os.path.join(imagepath_trash)

len(os.listdir(PATH_TRAIN))
IMG_HEIGHT = 32
IMG_WIDTH = 32

image_gen = ImageDataGenerator(rescale=1./255)

train_data_gen = image_gen.flow_from_directory(
    directory = train_dir, 
    shuffle=True, 
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

test_data_gen = image_gen.flow_from_directory(
    directory = test_dir, 
    shuffle=True, 
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(6, activation='softmax'))

model=Sequential([
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(6, activation='softmax')
])

batch_size = 38
epochs = 80
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# model.summary()

num_cardboard_train = len(os.listdir(imagepath_cardboard_dir))
num_glass_train = len(os.listdir(imagepath_glass_dir))
num_metal_train = len(os.listdir(imagepath_metal_dir))                  
num_paper_train = len(os.listdir(imagepath_cardboard_dir))
num_plastic_train = len(os.listdir(imagepath_glass_dir))
num_trash_train = len(os.listdir(imagepath_trash_dir))

num_cardboard_test = len(os.listdir(graypath_cardboard))
num_glass_test = len(os.listdir(graypath_glass))
num_metal_test = len(os.listdir(graypath_metal))
num_paper_test = len(os.listdir(graypath_paper))
num_plastic_test = len(os.listdir(graypath_plastic))
num_trash_test = len(os.listdir(graypath_trash))

total_train = num_cardboard_train + num_glass_train + num_metal_train + num_paper_train + num_plastic_train + num_trash_train
total_test = num_cardboard_test + num_glass_test + num_metal_test + num_paper_test + num_plastic_test + num_trash_test

print(total_train)
history = model.fit(
        train_data_gen,
        validation_data = train_data_gen,
        steps_per_epoch = total_train // batch_size,
        epochs = epochs,
        validation_steps= total_test // batch_size,
        callbacks = [tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.01,
                    patience=7)]
    )                   