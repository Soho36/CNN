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
    image_size=(300, 300),
    batch_size=32,
    label_mode='binary',  # Binary classification (Buy = 1, Sell = 0)
    color_mode='rgb'
)

# Load test dataset
test_dataset = image_dataset_from_directory(
    dataset_path + "/test",
    image_size=(300, 300),
    batch_size=32,
    label_mode='binary',  # Binary classification
    color_mode='rgb'
)

# Normalize pixel values (0-1 range)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Define CNN model
model = models.Sequential([
    layers.Input(shape=(300, 300, 3)),  # Explicit Input layer

    # Adding L2 regularization to Conv2D layers
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),

    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dropout(0.1),  # Dropout for regularization
    layers.Dense(1, activation='sigmoid')  # 1 output neuron for binary classification
])

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

# Plot accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


# Prediction function with threshold
# def predict_trade(image_path, threshold=0.6):
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(300, 300))
#     img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     img_array = tf.expand_dims(img_array, axis=0)
#
#     prediction = model.predict(img_array)[0][0]  # Sigmoid output
#
#     if prediction > threshold:
#         return "Buy"
#     elif prediction < (1 - threshold):
#         return "Sell"
#     else:
#         return "Undefined"


# # Example usage
# image_path = "path/to/test/image.jpg"
# print(f"Prediction: {predict_trade(image_path)}")
