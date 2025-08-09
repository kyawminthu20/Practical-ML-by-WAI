
from tensorflow.keras.models import load_model
import os
import json
from tensorflow.keras.preprocessing import image
import numpy as np

class CatAndDogModel:

    def __init__(self):
        self.model_path = os.getcwd() + "/ml_models/cat_and_dog_model.h5"
        self.class_path = os.getcwd() + "/ml_models/class_names.json"
        self.input_img_size = (128,128)  # Assuming the model expects 224x224 images
        self.model = None
        self.class_indices = None


    def load_model(self):
        print("Loading model and class indices...")
        try:
            self.model = load_model(self.model_path)
    
            with open(self.class_path, 'r') as f:
                self.class_indices = json.load(f)

            print("Model and class indices loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model or class indices: {e}")
            return False
 
        return True
    

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=self.input_img_size)
        img_array = image.img_to_array(img)  # shape: (_, _, 3)
        img_array = img_array / 255.0        # normalize

        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, _, _, 3)

 
        return img_array


    def predict(self, img_array):
        prediction = self.model.predict(img_array)[0][0]
        class_name = "dogs" if prediction > 0.5 else "cats"
        return class_name, prediction
