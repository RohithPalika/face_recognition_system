import os
import cv2
import numpy as np
import time

from .detector import FaceDetectorYuNet
from .recognizer import FaceRecognizerSFace
from .age_gender import AgeGenderEstimator
from .emotion import EmotionClassifier
from .tracker import MultiFaceTracker
from .visualizer import BiometricVisualizer
from .logger import AttendanceLogger

class BiometricPipeline:
    """Unified pipeline connecting Detection, Recognition, Demographics, Emotion, and HUD."""

    def __init__(self, 
                 models_dir="models", 
                 db_dir="known_faces",
                 score_threshold=0.6):
        
        self.detector = FaceDetectorYuNet(
            model_path=os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"),
            score_threshold=score_threshold
        )
        self.recognizer = FaceRecognizerSFace(
            model_path=os.path.join(models_dir, "face_recognition_sface_2021dec.onnx"),
            db_dir=db_dir
        )
        self.age_gender = AgeGenderEstimator(
            age_proto="age_deploy.prototxt",
            age_model="age_net.caffemodel",
            gender_proto="gender_deploy.prototxt",
            gender_model="gender_net.caffemodel"
        )
        self.emotion_clf = EmotionClassifier(
            model_path=os.path.join(models_dir, "emotion-ferplus-8.onnx")
        )
        self.tracker = MultiFaceTracker()
        self.visualizer = BiometricVisualizer()
        self.logger = AttendanceLogger()

        self.last_time = time.time()
        self.fps = 0.0

    def process_frame(self, frame, log_attendance=True):
        """
        Runs the full end-to-end biometric analysis on a single video frame or static image.
        Returns:
            annotated_frame: Frame with Futuristic Biometric HUD overlay
            tracked_faces: list of TrackedFace objects with all predictions
            detected_data: list of raw detection dictionaries
        """
        if frame is None:
            return None, [], []

        # Calculate FPS
        curr_time = time.time()
        dt = curr_time - self.last_time
        self.last_time = curr_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps

        h, w = frame.shape[:2]
        detections = self.detector.detect(frame)
        current_frame_data = []

        for det in detections:
            box = det['box']
            raw_face = det['raw_face']
            landmarks = det['landmarks']
            
            x, y, bw, bh = box
            # Add padding for age/gender crop
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)
            face_crop = frame[y1:y2, x1:x2]

            # 1. Feature Extraction & Recognition
            feature, aligned_face = self.recognizer.extract_feature(frame, raw_face)
            name, name_conf, is_match = self.recognizer.recognize(feature)

            # 2. Continuous Age & Gender
            demo = self.age_gender.predict(face_crop)
            age = demo['age_exact']
            gender = demo['gender']

            # 3. Emotion Analysis
            emo = self.emotion_clf.predict(aligned_face)
            emotion = emo['emotion']

            # 4. Attendance / Audit Logging
            if log_attendance and is_match:
                self.logger.log_detection(name, name_conf, age, gender, emotion)

            current_frame_data.append({
                'box': box,
                'raw_face': raw_face,
                'landmarks': landmarks,
                'age': age,
                'gender': gender,
                'emotion': emotion,
                'name': name,
                'name_conf': name_conf,
                'demo_full': demo,
                'emo_full': emo
            })

        # Update temporal tracker and smoothing
        tracked_faces = self.tracker.update(current_frame_data)

        # Render Futuristic HUD
        annotated_frame = self.visualizer.render_hud(frame, tracked_faces, fps=self.fps)

        return annotated_frame, tracked_faces, current_frame_data

    def enroll_from_image(self, name, image):
        """Enrolls a person into database given a photo."""
        detections = self.detector.detect(image)
        if not detections:
            return False, "No face detected in the provided image."
        # Pick largest face
        largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        return self.recognizer.enroll_face(name, image, largest['raw_face'])
