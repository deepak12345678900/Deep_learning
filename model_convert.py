import tensorflow as tf

# Load the HDF5 model
h5_model_path = 'cats_model.h5'  # Replace with the path to your HDF5 model file
loaded_model = tf.keras.models.load_model(h5_model_path)

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_model_path = 'cats_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model converted from HDF5 and saved to: {tflite_model_path}')
