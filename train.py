# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # Number of times to go through the dataset
)

# Save the model
model.save('candlestick_cnn_model.h5')
