#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:22:28 2023

@author: kmark7
"""

import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow_hub as hub

splitfolders.ratio("images", output="dogs", seed=1337, ratio=(.6, .4), group_prefix=None, move=True)

IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32
train_dir = "dogs/train/"
test_dir = "dogs/val/"

train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=40,
    shear_range=0.3,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1/255.)

print("Training images:")
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)
print("Testing images:")
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

data_dir = pathlib.Path("dogs/train")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

#resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5"
pro_model_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
resnet = hub.KerasLayer(pro_model_url, trainable=False, name='EfficientNet', input_shape=IMAGE_SHAPE+(3,))
model = tf.keras.Sequential([
    resnet,
    layers.Dense(420, activation='softmax', name='output_layer', kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

model.summary()

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=768, decay_rate=0.9, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = EarlyStopping(
    min_delta=0.01,
    patience=5,
    restore_best_weights=True,
)

#---------- OPCIONÁLIS RÉSZ 
train_datagen_augmented = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=40,
    shear_range=0.3,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

augmented_train_data = train_datagen_augmented.flow_from_directory(
    train_dir,
    target_size=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

combined_train_data = tf.data.Dataset.from_generator(
    lambda: train_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 420), dtype=tf.float32)
    )
).concatenate(
    tf.data.Dataset.from_generator(
        lambda: augmented_train_data,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 420), dtype=tf.float32)
        )
    )
)
#----------

history = model.fit(
    combined_train_data, #combined_train_data => ha adatbővítést is szeretnénk
    epochs=15,
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=len(test_data),
    callbacks=[early_stopping],
)

def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

plot_loss_curves(history)

model.save('KutyaJoModel')
model2 = tf.keras.models.load_model("KutyaJoModel")

def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = tf.expand_dims(img, axis=0)
    img = img / 255.
    return img

image = load_and_prep_image("elsoTeszt.JPG")
pred = model2.predict(image)
pred_class = class_names[np.argmax(pred)]

print(pred_class)
