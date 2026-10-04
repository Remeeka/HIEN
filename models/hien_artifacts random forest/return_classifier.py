import joblib
import json
import pandas as pd
import numpy as np

class HIENReturnClassifier:
    def __init__(self, model_path='return_classifier.joblib', mapping_path='class_mapping.json'):
        self.pipeline = joblib.load(model_path)
        with open(mapping_path, 'r') as f:
            raw_mapping = json.load(f)
            self.mapping = {int(k): v for k, v in raw_mapping.items()}

    def predict(self, data_dict):
        df_input = pd.DataFrame([data_dict])
        probs = self.pipeline.predict_proba(df_input)[0]
        pred_idx = np.argmax(probs)
        prediction = self.mapping[pred_idx]
        
        # Review recommendation check
        confidence = probs[pred_idx]
        review_recommended = confidence < 0.65
        
        probabilities_dict = {self.mapping[i]: float(probs[i]) for i in range(len(probs))}
        
        return {
            "prediction": prediction,
            "probabilities": probabilities_dict,
            "review_recommended": bool(review_recommended)
        }
