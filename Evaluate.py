# Evaluate the model on validation data
loss, accuracy = model.evaluate(val_data)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Load a single image for prediction
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'path_to_new_chart.png'
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict (output will be close to 0 for Sell, 1 for Buy)
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("Buy")
else:
    print("Sell")
