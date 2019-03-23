import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import keras
from keras.layers import Dense , GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


base_model = MobileNet(weights = 'imagenet' , include_top = False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation = 'relu')(x)

model = Model(inputs = base_model.input,output = preds)

for i ,layer in enumerate (model.layers):
    print(i,layer.name)
    
for layer in model.layers:
    layer.trainable = False
    
for layer in model.layers[:20] :
    layer.trainable = False
    
for layer in model.layers[20:]:
    layer.trainable = True
    
    
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator=train_datagen.flow_from_directory('path-to-tht-main-data-folder',
                                                  target_size=(224,224),
                                                  color_mode ='rgb',
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=True)

model.compile(optimizer = 'Adam',loss='categorical_crossentropy',metrices=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator = train_generator, 
                     steps_per_epoch=step_size_train,
                     epoch=10)
    
    