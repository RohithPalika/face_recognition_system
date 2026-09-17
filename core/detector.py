import os
import cv2
import numpy as np

class FaceDetectorYuNet:
    """High-accuracy real-time face detector using OpenCV YuNet ONNX model."""
    
    def __init__(self, model_path="models/face_detection_yunet_2023mar.onnx", 
                 score_threshold=0.6, nms_threshold=0.3, top_k=5000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.input_size = (300, 300)
        
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self.input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )
        
    def detect(self, image):
        """
        Detects faces in an image.
        Returns:
            list of dicts containing:
                - 'box': [x, y, w, h]
                - 'score': confidence (0.0 to 1.0)
                - 'landmarks': [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5)]
                - 'raw_face': full raw face detection array for SFace
        """
        if image is None:
            return []
            
        h, w = image.shape[:2]
        if self.input_size != (w, h):
            self.input_size = (w, h)
            self.detector.setInputSize((w, h))
            
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
            
        results = []
        for face in faces:
            # face: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, bw, bh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[-1])
            
            # 5 landmarks
            landmarks = [
                (int(face[4]), int(face[5])),   # Right Eye
                (int(face[6]), int(face[7])),   # Left Eye
                (int(face[8]), int(face[9])),   # Nose Tip
                (int(face[10]), int(face[11])), # Right Mouth Corner
                (int(face[12]), int(face[13]))  # Left Mouth Corner
            ]
            
            results.append({
                "box": [max(0, x), max(0, y), bw, bh],
                "score": score,
                "landmarks": landmarks,
                "raw_face": face
            })
            
        return results
