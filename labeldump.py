from sklearn.preprocessing import LabelEncoder
import joblib
import os

dataset_path = 'imagery/realsense_overhead/' 
label_encoder = LabelEncoder()
dish_ids = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
label_encoder.fit(dish_ids)


joblib.dump(label_encoder, 'label_encoder.joblib')
