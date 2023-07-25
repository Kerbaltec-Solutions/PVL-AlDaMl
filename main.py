# This code was written by Björn Schnabel as part of a university project.
# This code is not and will not be maintained and should not be used outside of the university context
# 25.07.2023 21:40


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading the FER2013 dataset file
data = pd.read_csv('fer2013.csv')

# Splitting the data into features (pixels) and labels (emotions)
pixels = data['pixels'].tolist()
emotions = pd.get_dummies(data['emotion']).values

# Reshaping the pixels to be a 2D image
images = np.array([np.fromstring(pixel, dtype='uint8', sep=' ') for pixel in pixels])
images = images.reshape((-1, 48, 48, 1))  # Reshaping dimensions for CNN

# Mirroring the data and appending it on the unflipped data
images_flip=np.copy(images)
np.fliplr(images_flip)
np.append(images,images_flip)

# Splitting the data into training, validation and test sets
train_images, test_images, train_emotions, test_emotions = train_test_split(images, emotions, test_size=0.1, random_state=42)
validation_images, test_images, validation_emotions, test_emotions = train_test_split(train_images, train_emotions, test_size=0.2, random_state=42)

# Creating the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),  
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),  
    tf.keras.layers.UpSampling2D (size=(4,4)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.UpSampling2D (size=(8,8)),
    tf.keras.layers.Dropout(0.25),  
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(236, (3, 3), activation='relu'),
    tf.keras.layers.UpSampling2D (size=(4,4)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.UpSampling2D (size=(8,8)),
    tf.keras.layers.Dropout(0.25),  
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(2, (3, 3), activation='relu'),
    tf.keras.layers.UpSampling2D (size=(4,4)),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),  
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()

# Compiling the model
opti=tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])

# Adding early stopping to save computing time on models which do not work
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

# Training the model
history = model.fit(train_images, train_emotions, epochs=80, batch_size=64,
                    validation_data=(validation_images, validation_emotions),
                    callbacks=[early_stopping])

# Evaluating the model on the test set
_, accuracy = model.evaluate(test_images, test_emotions)
print('Test Accuracy:', accuracy)

# Plotting the training loss and validation accuracy over time
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Making predictions on the test set
predictions = model.predict(test_images)
predicted_emotions = np.argmax(predictions, axis=1)
expected_emotions = np.argmax(test_emotions, axis=1)

# Creating a scatter plot comparing predicted results with expected results
plt.figure(figsize=(6, 6))
plt.scatter(expected_emotions, predicted_emotions, alpha=0.5)
plt.xlabel('Expected Emotions')
plt.ylabel('Predicted Emotions')
plt.xticks(range(7), ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'], rotation=45)
plt.yticks(range(7), ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
plt.title('Predicted vs Expected Emotions')
plt.grid(True)
plt.show()
