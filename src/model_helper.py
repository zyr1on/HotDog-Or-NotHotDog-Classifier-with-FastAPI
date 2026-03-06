import os
import numpy as np
import keras
from PIL import Image
import io

class HotdogPredictor:
    def __init__(self, model_path='best_hotdog_model.keras'):
        self.img_size = 224
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found!: {self.model_path}")
        return keras.models.load_model(self.model_path)

    def predict_image(self, file_contents):
        # return bytes to image
        img = Image.open(io.BytesIO(file_contents)).convert('RGB')
        
        # preprocessing
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # predict
        prediction_score = self.model.predict(img_array, verbose=0)[0][0]
        
        # result definition
        label = "NOT HOT DOG" if prediction_score > 0.5 else "HOT DOG"
        
        return {
            "label": label,
            "score": float(prediction_score)
        }