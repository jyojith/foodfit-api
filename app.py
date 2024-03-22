from flask import Flask, request, jsonify
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import io
import json

def parse_metadata(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Splitting the line into components
            data = line.strip().split(',')
            # Extracting dish id and general dish information
            dish_id = data[0]
            dish_info = {
                'total_calories': float(data[1]),
                'total_mass': float(data[2]),
                'total_fat': float(data[3]),
                'total_carb': float(data[4]),
                'total_protein': float(data[5]),
                'ingredients': []
            }
            metadata[dish_id] = dish_info
            try:
                ingr_data = data[6:]  # Start from the 6th element for ingredients
                for i in range(0, len(ingr_data), 7):  # Process each ingredient (7 fields per ingredient)
                    ingr_info = {}
                    if i + 6 < len(ingr_data):  # Ensure there are enough fields for an ingredient
                        ingr_info = {
                            'id': ingr_data[i],
                            'name': ingr_data[i + 1],
                            'grams': float(ingr_data[i + 2]),
                            'calories': float(ingr_data[i + 3]),
                            'fat': float(ingr_data[i + 4]),
                            'carb': float(ingr_data[i + 5]),
                            'protein': float(ingr_data[i + 6])
                        }
                    dish_info['ingredients'].append(ingr_info)

                metadata[dish_id] = dish_info
            except ValueError:
                continue
    return metadata

model_path = '/Users/jyojith/starthack/nutrition5k_dataset_nosides/my_model.h5'  # Replace with your model path
model = load_model(model_path)

metadata_path = 'metadata/dish_metadata_cafe1.csv'
metadata = parse_metadata(metadata_path)

dataset_path = '/Users/jyojith/starthack/nutrition5k_dataset_nosides/imagery/realsense_overhead/'  # Replace with your dataset path
dish_ids = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
label_encoder = LabelEncoder()
label_encoder.fit(dish_ids)


app = Flask(__name__)

# Assuming the model, metadata, and label_encoder are initialized elsewhere and accessible
# from your_module import model, metadata, label_encoder

@app.route('/predict', methods=['POST'])

# def prepare_image(image_path, target_size=(150, 150)):
#     # Load the RGB image
#     rgb_img = img_to_array(load_img(image_path, target_size=target_size))
#
#     # Create mock depth color and depth raw images by replicating the RGB image
#     depth_color_img = rgb_img.copy()
#     depth_raw_img = rgb_img.copy()
#
#     # Combine the images into a single array
#     combined_img = np.concatenate([rgb_img, depth_color_img, depth_raw_img], axis=-1)
#     combined_img = np.expand_dims(combined_img, axis=0)  # Add batch dimension
#     return combined_img

def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_binary = file.read()
    json_response = process_image_and_return_json(image_binary)
    return jsonify(json_response), 200

def process_image_and_return_json(image_binary, top_n=1):
    def prepare_image(image, target_size=(150, 150)):
        img = img_to_array(image.resize(target_size))
        depth_color_img = img.copy()
        depth_raw_img = img.copy()
        combined_img = np.concatenate([img, depth_color_img, depth_raw_img], axis=-1)
        return np.expand_dims(combined_img, axis=0)

    image = Image.open(io.BytesIO(image_binary))
    prepared_image = prepare_image(image)

    probabilities = model.predict(prepared_image)[0]
    dish_id_probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}
    sorted_dishes = sorted(dish_id_probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]

    response = []
    for dish_id, probability in sorted_dishes:
        dish_info = {
            'dish_id': dish_id,
            'probability': float(probability),
            'metadata': metadata.get(dish_id, 'No metadata available for this dish')
        }
        response.append(dish_info)

    return { 'result': response}

if __name__ == '__main__':
    app.run(debug=True)
