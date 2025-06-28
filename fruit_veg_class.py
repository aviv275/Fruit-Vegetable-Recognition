# %%
# Fruit and Vegetable Classification Model
# Adapted for local project structure

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# %%
# Create a list with the filepaths for training, testing, and validation
# Updated paths to match local project structure
train_dir = Path('input/train')
test_dir = Path('input/test')
val_dir = Path('input/validation')

# Get all image files with various extensions
train_filepaths = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    train_filepaths.extend(list(train_dir.glob(f'**/{ext}')))

test_filepaths = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    test_filepaths.extend(list(test_dir.glob(f'**/{ext}')))

val_filepaths = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    val_filepaths.extend(list(val_dir.glob(f'**/{ext}')))

print(f"Found {len(train_filepaths)} training images")
print(f"Found {len(test_filepaths)} test images")
print(f"Found {len(val_filepaths)} validation images")

# %%
def image_processing(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
    """
    labels = [str(filepath[i]).split("/")[-2] \
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop = True)
    
    return df

# %%
train_df = image_processing(train_filepaths)
test_df = image_processing(test_filepaths)
val_df = image_processing(val_filepaths)

# %%
print('-- Training set --\n')
print(f'Number of pictures: {train_df.shape[0]}\n')
print(f'Number of different labels: {len(train_df.Label.unique())}\n')
print(f'Labels: {sorted(train_df.Label.unique())}')

# %%
train_df.head(5)

# %%
# Create a DataFrame with one Label of each category
df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()

# Display some pictures of the dataset
fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(12, 10),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    if i < len(df_unique):
        ax.imshow(plt.imread(df_unique.Filepath[i]))
        ax.set_title(df_unique.Label[i], fontsize = 10)
    else:
        ax.axis('off')
plt.tight_layout(pad=0.5)
plt.show()

# %%
# Data generators with proper preprocessing
train_generator = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_generator = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# %%
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)

# %%
val_images = train_generator.flow_from_dataframe(
    dataframe=val_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)

# %%
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# %%
# Get the number of classes dynamically
num_classes = len(train_images.class_indices)
print(f"Number of classes: {num_classes}")

# %%
pretrained_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False

# %%
inputs = pretrained_model.input

x = Dense(128, activation='relu')(pretrained_model.output)
x = Dense(128, activation='relu')(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# %%
history = model.fit(
    train_images,
    validation_data=val_images,
    batch_size=32,
    epochs=10,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

# %%
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_images)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# %%
# Predict the label of the test_images
pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)

# Map the label
labels = train_images.class_indices
labels = dict((v, k) for k, v in labels.items())
pred1 = [labels[k] for k in pred]

# Display some predictions
print("Sample predictions:")
for i in range(min(10, len(pred1))):
    print(f"Predicted: {pred1[i]}")

# %%
def predict_single_image(location):
    """Predict a single image"""
    img = load_img(location, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res

# %%
# Test prediction on a sample image
if len(test_filepaths) > 0:
    sample_image = str(test_filepaths[0])
    print(f"Testing prediction on: {sample_image}")
    prediction = predict_single_image(sample_image)
    print(f"Predicted: {prediction}")

# %%
# Save the model
model.save('FV2.h5')
print("Model saved as 'FV.h5'")

# %%
# Save class indices for use in the app
import json
with open('class_indices.json', 'w') as f:
    json.dump(train_images.class_indices, f)
print("Class indices saved as 'class_indices.json'")

# %%



