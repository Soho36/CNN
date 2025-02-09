import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Image parameters
img_width, img_height = 100, 200  # Adjust to your image size
batch_size = 32  # Number of images per training step

# Paths to your dataset
data_dir = 'data'

# Training and validation split
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    validation_split=0.2  # 20% of data for validation
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # 'binary' for Buy/Sell
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)
