# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from keras import datasets, layers, models

# %%
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# %%
x_train.shape

# %%
x_test.shape

# %%
x_train[0]

# %%
y_train.shape

# %%
y_train[:5]

# %%
plt.imshow(x_train[0])

# %% [markdown]
# Normalization of x_train and x_test

# %%
x_train = x_train/255
x_test = x_test/255

# %%
x_train[0]

# %% [markdown]
# **Prediction Using ANN**

# %%
ann = models.Sequential([
        layers.Flatten(input_shape=(28,28)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(x_train, y_train, epochs=5)

# %%
ann.evaluate(x_test, y_test)

# %%
y_pred = ann.predict(x_test)    

# %%
y_pred[0]

# %%
np.argmax(y_pred[0])

# %%
plt.imshow(x_test[0])

# %%
y_predicted_labels = [np.argmax(i) for i in y_pred]

# %%
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm

# %%
import seaborn as sn
plt.figure(figsize = (9,5))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# %%
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# %%
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
cnn.fit(x_train, y_train, epochs=10)

# %%
cnn.evaluate(x_test,y_test)

# %%



