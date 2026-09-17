import os
import json
import cv2
import numpy as np

class FaceRecognizerSFace:
    """Deep learning Face Recognition using OpenCV SFace ONNX model with database management."""
    
    COSINE_THRESHOLD = 0.363  # OpenCV official threshold for SFace cosine similarity
    HIGH_CONF_THRESHOLD = 0.45

    def __init__(self, model_path="models/face_recognition_sface_2021dec.onnx", 
                 db_dir="known_faces"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        self.model_path = model_path
        self.db_dir = db_dir
        self.db_file = os.path.join(db_dir, "database.json")
        os.makedirs(db_dir, exist_ok=True)
        
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config=""
        )
        
        # In-memory dictionary: name -> list of feature vectors
        self.database = {}
        self.load_database()

    def extract_feature(self, image, raw_face):
        """Aligns face and extracts 128D feature embedding."""
        aligned_face = self.recognizer.alignCrop(image, raw_face)
        feature = self.recognizer.feature(aligned_face)
        return feature, aligned_face

    def enroll_face(self, name, image, raw_face):
        """Enrolls a person into the database with a feature vector and reference photo."""
        feature, aligned_face = self.extract_feature(image, raw_face)
        
        # Save reference thumbnail
        clean_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
        face_img_path = os.path.join(self.db_dir, f"{clean_name}.jpg")
        cv2.imwrite(face_img_path, aligned_face)
        
        feat_list = feature.flatten().tolist()
        if clean_name not in self.database:
            self.database[clean_name] = []
        self.database[clean_name].append(feat_list)
        
        self.save_database()
        return True, f"Successfully enrolled {clean_name}"

    def recognize(self, feature, threshold=COSINE_THRESHOLD):
        """
        Matches a feature vector against the database.
        Returns: (best_name, confidence_score, is_match)
        """
        if not self.database or feature is None:
            return "Unknown", 0.0, False

        best_name = "Unknown"
        best_score = -1.0

        for name, feat_list in self.database.items():
            for stored_feat in feat_list:
                ref_feat = np.array(stored_feat, dtype=np.float32).reshape(1, 128)
                score = self.recognizer.match(feature, ref_feat, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = float(score)
                    if score >= threshold:
                        best_name = name

        is_match = best_score >= threshold
        # Normalize score to a percentage 0-100% for display
        # Cosine score typically ranges from -1 to 1, with matches > 0.36
        display_conf = max(0.0, min(100.0, (best_score + 0.2) / 1.2 * 100))
        return best_name, display_conf, is_match

    def save_database(self):
        """Saves known faces database to JSON file."""
        with open(self.db_file, "w") as f:
            json.dump(self.database, f, indent=2)

    def load_database(self):
        """Loads database from JSON file."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    self.database = json.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
                self.database = {}
        else:
            self.database = {}

    def get_enrolled_names(self):
        """Returns list of enrolled person names."""
        return list(self.database.keys())

    def delete_person(self, name):
        """Deletes a person from the database."""
        if name in self.database:
            del self.database[name]
            self.save_database()
            # Remove photo if exists
            photo_path = os.path.join(self.db_dir, f"{name}.jpg")
            if os.path.exists(photo_path):
                try:
                    os.remove(photo_path)
                except Exception:
                    pass
            return True
        return False
