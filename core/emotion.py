import os
import cv2
import numpy as np

class EmotionClassifier:
    """Facial Expression & Emotion Recognition using FER+ ONNX deep model."""

    EMOTIONS = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt']
    EMOJI_MAP = {
        'Neutral': '😐',
        'Happy': '😊',
        'Surprise': '😲',
        'Sad': '😢',
        'Angry': '😠',
        'Disgust': '🤢',
        'Fear': '😨',
        'Contempt': '😏'
    }

    def __init__(self, model_path="models/emotion-ferplus-8.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def predict(self, face_img):
        """
        Predicts facial emotion from aligned face image.
        Returns:
            dict containing:
                - 'emotion': dominant emotion string (e.g., 'Happy')
                - 'emoji': emoji icon for the emotion (e.g., '😊')
                - 'confidence': confidence percentage (0.0 to 100.0)
                - 'scores': dictionary mapping each emotion to its probability
        """
        if face_img is None or face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
            return {
                "emotion": "Neutral",
                "emoji": "😐",
                "confidence": 50.0,
                "scores": {e: 1.0 / len(self.EMOTIONS) for e in self.EMOTIONS}
            }

        # Convert to Grayscale & resize to 64x64 for FER+
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        resized = cv2.resize(gray, (64, 64))
        blob = resized.astype(np.float32).reshape(1, 1, 64, 64)

        self.net.setInput(blob)
        preds = self.net.forward()[0]
        probs = self._softmax(preds)

        top_idx = int(np.argmax(probs))
        dominant_emotion = self.EMOTIONS[top_idx]
        confidence = float(probs[top_idx] * 100)

        score_dict = {
            emotion: round(float(probs[i]), 3)
            for i, emotion in enumerate(self.EMOTIONS)
        }

        return {
            "emotion": dominant_emotion,
            "emoji": self.EMOJI_MAP.get(dominant_emotion, "😐"),
            "confidence": round(confidence, 1),
            "scores": score_dict
        }
