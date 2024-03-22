from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model('/Users/jyojith/starthack/nutrition5k_dataset_nosides/my_model.h5')


# Assuming you have a model defined and compiled

# Convert the model
full_model = tf.function(lambda x: model(x))
concrete_function = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
