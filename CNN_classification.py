import tensorflow as tf
import zipfile, os
import split_folders
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_preprocessing
from keras_preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

!wget --no-check-certificate \
  https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip
  
local_zip = '/tmp/rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
ds_dir = '/tmp/rockpaperscissors'
os.listdir('/tmp/rockpaperscissors/rps-cv-images')

split_folders.ratio('/tmp/rockpaperscissors/rps-cv-images', output="splitOutput", ratio=(0.6, 0.2, 0.2))
new_dir = 'splitOutput'

train_dir = os.path.join(new_dir, 'train')
train_rock_dir = os.path.join(train_dir, 'rock')
train_paper_dir = os.path.join(train_dir, 'paper')
train_scissors_dir = os.path.join(train_dir, 'scissors')

validation_dir = os.path.join(new_dir, 'val'
validation_rock_dir = os.path.join(validation_dir, 'rock')
validation_paper_dir = os.path.join(validation_dir, 'paper')
validation_scissors_dir = os.path.join(validation_dir, 'scissors')


train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    width_shift_range=0.2,
                    shear_range = 0.2,
                    brightness_range = [0.2,1.0],
                    zoom_range = [0.5,1.0],
                    fill_mode = 'nearest')
 
test_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    width_shift_range=0.2,
                    shear_range = 0.2,
                    brightness_range = [0.2,1.0],
                    zoom_range = [0.5,1.0],
                    fill_mode = 'nearest')
                    
train_generator = train_datagen.flow_from_directory(
        train_dir,  
        target_size=(150, 150),
        class_mode='categorical')
print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
        validation_dir, 
        target_size=(150, 150),
        class_mode='categorical')
print(validation_generator.class_indices)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()
model.compile(optimizer='SGD', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
history = model.fit(
      train_generator,
      steps_per_epoch=25,  
      epochs=15,
      validation_data=validation_generator, 
      validation_steps=5,  
      verbose=2)
