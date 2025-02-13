import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Define dataset path (relative or absolute)
dataset_path = "dataset"  # If in the same directory as script

# Load training dataset
train_dataset = image_dataset_from_directory(
    dataset_path + "/train",
    image_size=(300, 300),  # Resize to match your images
    batch_size=32,
    label_mode='int',  # Labels are automatically assigned based on folder names
    color_mode='rgb'
)

# Load test dataset
test_dataset = image_dataset_from_directory(
    dataset_path + "/test",
    image_size=(300, 300),
    batch_size=32,
    label_mode='int',  # Labels are automatically assigned based on folder names
    color_mode='rgb'
)

# Normalize pixel values (0-1 range)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# model = models.Sequential([
#     layers.Input(shape=(300, 300, 3)),  # Explicit Input layer
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),

#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(3)  # 3 classes: Buy, Sell, Hold
# ])

"""
Added regularization and dropout
"""
model = models.Sequential([
    layers.Input(shape=(300, 300, 3)),  # Explicit Input layer

    # Adding L2 regularization to Conv2D layers
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    layers.Flatten(),
    # Adding L2 regularization and dropout to Dense layers
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dropout(0.1),  # Drop 50% of neurons to prevent overfitting
    layers.Dense(3)  # 3 classes: Buy, Sell, Hold
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=test_dataset
)

"""
Evaluate the model
"""

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()
