import os
import cv2
import numpy as np

class AgeGenderEstimator:
    """Continuous Age and Gender Estimation using Deep Neural Networks."""

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-19)', '(20-32)', '(38-43)', '(48-53)', '(60-100)']
    AGE_MIDPOINTS = np.array([1.0, 5.0, 10.0, 17.0, 26.0, 40.5, 50.5, 75.0], dtype=np.float32)
    GENDER_CLASSES = ['Male', 'Female']

    def __init__(self, 
                 age_proto="age_deploy.prototxt", 
                 age_model="age_net.caffemodel",
                 gender_proto="gender_deploy.prototxt", 
                 gender_model="gender_net.caffemodel"):
        
        for path in [age_proto, age_model, gender_proto, gender_model]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

        self.age_net = cv2.dnn.readNet(age_model, age_proto)
        self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict(self, face_img):
        """
        Predicts continuous age and gender probability from a cropped face image.
        Returns:
            dict containing:
                - 'age_exact': float estimated continuous age (e.g. 24.3)
                - 'age_display': string formatted age (e.g. "24 yrs")
                - 'age_bucket': discrete bucket name
                - 'gender': "Male" or "Female"
                - 'gender_conf': float (0.0 - 100.0)
                - 'gender_probs': {'Male': float, 'Female': float}
        """
        if face_img is None or face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
            return {
                "age_exact": 25.0,
                "age_display": "25 yrs",
                "age_bucket": "(20-32)",
                "gender": "Unknown",
                "gender_conf": 50.0,
                "gender_probs": {"Male": 0.5, "Female": 0.5}
            }

        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227), 
            self.MODEL_MEAN_VALUES, swapRB=False
        )

        # Gender prediction
        self.gender_net.setInput(blob)
        gender_raw = self.gender_net.forward()[0]
        gender_probs = self._softmax(gender_raw)
        gender_idx = int(np.argmax(gender_probs))
        gender = self.GENDER_CLASSES[gender_idx]
        gender_conf = float(gender_probs[gender_idx] * 100)

        # Continuous Age prediction using expected value over softmax distribution
        self.age_net.setInput(blob)
        age_raw = self.age_net.forward()[0]
        age_probs = self._softmax(age_raw)
        
        # Expected continuous age = sum(prob_i * midpoint_i)
        expected_age = float(np.sum(age_probs * self.AGE_MIDPOINTS))
        bucket_idx = int(np.argmax(age_probs))
        age_bucket = self.AGE_BUCKETS[bucket_idx]

        return {
            "age_exact": round(expected_age, 1),
            "age_display": f"{int(round(expected_age))} yrs",
            "age_bucket": age_bucket,
            "gender": gender,
            "gender_conf": round(gender_conf, 1),
            "gender_probs": {
                "Male": round(float(gender_probs[0]), 3),
                "Female": round(float(gender_probs[1]), 3)
            }
        }
