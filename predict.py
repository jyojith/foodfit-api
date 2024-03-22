import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import joblib

model_path = '/Users/jyojith/starthack/nutrition5k_dataset_nosides/my_model.h5'  # Replace with your model path
model = load_model(model_path)

label_encoder = joblib.load('label_encoder.joblib')

import pandas as pd

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



def prepare_image(image_path, target_size=(150, 150)):
    # Load the RGB image
    rgb_img = img_to_array(load_img(image_path, target_size=target_size))

    # Create mock depth color and depth raw images by replicating the RGB image
    depth_color_img = rgb_img.copy()
    depth_raw_img = rgb_img.copy()

    # Combine the images into a single array
    combined_img = np.concatenate([rgb_img, depth_color_img, depth_raw_img], axis=-1)
    combined_img = np.expand_dims(combined_img, axis=0)  # Add batch dimension
    return combined_img




image_path = '/Users/jyojith/starthack/nutrition5k_dataset_nosides/pie.webp'  # Replace with your test image path
prepared_image = prepare_image(image_path, target_size=(150, 150))

# Load metadata
metadata_path = 'metadata/dish_metadata_cafe1.csv'
metadata = parse_metadata(metadata_path)

probabilities = model.predict(prepared_image)[0]
dish_id_probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}

top_n = 5
sorted_dishes = sorted(dish_id_probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]

response = []

for dish_id, probability in sorted_dishes:
    dish_info = {
        'dish_id': dish_id,
        'probability': float(probability)
    }

    if dish_id in metadata:
        dish_info['metadata'] = metadata[dish_id]
    else:
        dish_info['metadata'] = 'No metadata available for this dish.'

    response.append(dish_info)

# Convert the response to JSON format
json_response = json.dumps(response, indent=4)

# Output the JSON response
print(json_response)
