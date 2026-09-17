import numpy as np

class TrackedFace:
    """Represents a single persistent tracked face across video frames."""

    def __init__(self, track_id, box, raw_face, landmarks, age, gender, emotion, name, name_conf):
        self.track_id = track_id
        self.box = np.array(box, dtype=np.float32)  # [x, y, w, h]
        self.raw_face = raw_face
        self.landmarks = landmarks
        
        # Smoothed attributes
        self.age = float(age)
        self.gender = gender
        self.emotion = emotion
        self.name = name
        self.name_conf = float(name_conf)
        
        # History buffers for voting/smoothing
        self.age_history = [self.age]
        self.gender_history = [gender]
        self.emotion_history = [emotion]
        self.name_history = [name]
        
        self.disappeared_count = 0
        self.total_visible_frames = 1

    def update(self, box, raw_face, landmarks, age, gender, emotion, name, name_conf, alpha=0.55):
        """Updates tracked face with new frame observations using EMA smoothing."""
        # Box EMA smoothing
        self.box = alpha * np.array(box, dtype=np.float32) + (1.0 - alpha) * self.box
        self.raw_face = raw_face
        self.landmarks = landmarks

        # Age EMA smoothing
        self.age = alpha * float(age) + (1.0 - alpha) * self.age
        self.age_history.append(float(age))
        if len(self.age_history) > 15:
            self.age_history.pop(0)

        # Gender & Emotion history (majority vote smoothing)
        self.gender_history.append(gender)
        if len(self.gender_history) > 10:
            self.gender_history.pop(0)
        self.gender = max(set(self.gender_history), key=self.gender_history.count)

        self.emotion_history.append(emotion)
        if len(self.emotion_history) > 10:
            self.emotion_history.pop(0)
        self.emotion = max(set(self.emotion_history), key=self.emotion_history.count)

        # Name / Recognition smoothing
        self.name_history.append(name)
        if len(self.name_history) > 10:
            self.name_history.pop(0)
        
        # If any recent frame confirmed identity with high confidence, prefer the identified name
        identified_names = [n for n in self.name_history if n != "Unknown"]
        if identified_names:
            self.name = max(set(identified_names), key=identified_names.count)
        else:
            self.name = "Unknown"

        self.name_conf = alpha * float(name_conf) + (1.0 - alpha) * self.name_conf
        self.disappeared_count = 0
        self.total_visible_frames += 1


class MultiFaceTracker:
    """Centroid & IOU based multi-face tracker with temporal attribute smoothing."""

    def __init__(self, max_disappeared=10, distance_threshold=80.0):
        self.next_track_id = 1
        self.tracks = {}  # track_id -> TrackedFace
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def _get_centroid(self, box):
        x, y, w, h = box
        return np.array([x + w / 2.0, y + h / 2.0])

    def update(self, detected_faces_data):
        """
        Takes a list of face observation dicts for the current frame:
        [{
            'box': [x, y, w, h],
            'raw_face': ...,
            'landmarks': [...],
            'age': float,
            'gender': str,
            'emotion': str,
            'name': str,
            'name_conf': float
        }, ...]

        Returns:
            list of TrackedFace objects for rendering.
        """
        if len(detected_faces_data) == 0:
            # Mark all existing tracks as disappeared
            to_delete = []
            for track_id, track in self.tracks.items():
                track.disappeared_count += 1
                if track.disappeared_count > self.max_disappeared:
                    to_delete.append(track_id)
            for tid in to_delete:
                del self.tracks[tid]
            return []

        if len(self.tracks) == 0:
            # Register all new detections
            for data in detected_faces_data:
                track = TrackedFace(
                    track_id=self.next_track_id,
                    box=data['box'],
                    raw_face=data['raw_face'],
                    landmarks=data['landmarks'],
                    age=data['age'],
                    gender=data['gender'],
                    emotion=data['emotion'],
                    name=data['name'],
                    name_conf=data['name_conf']
                )
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
            return list(self.tracks.values())

        # Match existing tracks to detections using centroid distance
        track_ids = list(self.tracks.keys())
        track_centroids = [self._get_centroid(self.tracks[tid].box) for tid in track_ids]
        det_centroids = [self._get_centroid(d['box']) for d in detected_faces_data]

        # Calculate distance matrix
        D = np.zeros((len(track_centroids), len(det_centroids)))
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(det_centroids):
                D[i, j] = np.linalg.norm(tc - dc)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.distance_threshold:
                continue

            track_id = track_ids[row]
            data = detected_faces_data[col]
            self.tracks[track_id].update(
                box=data['box'],
                raw_face=data['raw_face'],
                landmarks=data['landmarks'],
                age=data['age'],
                gender=data['gender'],
                emotion=data['emotion'],
                name=data['name'],
                name_conf=data['name_conf']
            )

            used_rows.add(row)
            used_cols.add(col)

        # Check unmatched tracks
        unused_rows = set(range(len(track_ids))) - used_rows
        for row in unused_rows:
            track_id = track_ids[row]
            self.tracks[track_id].disappeared_count += 1

        # Delete expired tracks
        to_delete = [tid for tid, track in self.tracks.items() if track.disappeared_count > self.max_disappeared]
        for tid in to_delete:
            del self.tracks[tid]

        # Register unmatched new detections
        unused_cols = set(range(len(detected_faces_data))) - used_cols
        for col in unused_cols:
            data = detected_faces_data[col]
            track = TrackedFace(
                track_id=self.next_track_id,
                box=data['box'],
                raw_face=data['raw_face'],
                landmarks=data['landmarks'],
                age=data['age'],
                gender=data['gender'],
                emotion=data['emotion'],
                name=data['name'],
                name_conf=data['name_conf']
            )
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        return [t for t in self.tracks.values() if t.disappeared_count == 0]
