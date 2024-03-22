import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

dataset_path = '/Users/jyojith/starthack/nutrition5k_dataset_nosides/imagery/realsense_overhead/'  # Replace with your dataset path
image_size = (150, 150)  # Replace with your image size

# Lists to store processed images and labels
images = []
labels = []

# Iterate through each dish folder
for dish_folder in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, dish_folder)):
        # Construct paths for each image type
        depth_color_path = os.path.join(dataset_path, dish_folder, 'depth_color.png')
        depth_raw_path = os.path.join(dataset_path, dish_folder, 'depth_raw.png')
        rgb_path = os.path.join(dataset_path, dish_folder, 'rgb.png')

        # Load and process each image
        try:
            depth_color_img = img_to_array(load_img(depth_color_path, target_size=image_size))
            depth_raw_img = img_to_array(load_img(depth_raw_path, target_size=image_size))
            rgb_img = img_to_array(load_img(rgb_path, target_size=image_size))
        except Exception as e:
            print(e)
            print(depth_raw_path)

        # Combine images into a single array (you can also stack or concatenate depending on your model design)
        combined_img = np.concatenate([depth_color_img, depth_raw_img, rgb_img], axis=-1)

        # Append the combined image and label to the lists
        images.append(combined_img)
        labels.append(dish_folder)  # Assuming folder name is the label

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2)

num_classes = len(np.unique(encoded_labels))

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 9)),  # Adjust the input shape (9 channels for 3 images)
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')  # num_classes should be the number of dish categories
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save('my_model.h5')
# Save the label encoder (you can use joblib or pickle)
