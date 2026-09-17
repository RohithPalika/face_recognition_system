import cv2
import numpy as np
import time
import math

class BiometricVisualizer:
    """Futuristic Sci-Fi Biometric HUD Overlay Visualizer for Face Recognition."""

    # Colors in BGR
    COLOR_CYAN = (245, 215, 0)      # Identified Known Person
    COLOR_AMBER = (0, 165, 255)     # Scanning / Unknown
    COLOR_NEON_RED = (60, 60, 255)  # Alert / Unknown
    COLOR_WHITE = (240, 240, 240)
    COLOR_DARK = (18, 18, 22)
    COLOR_ACCENT = (255, 100, 0)

    def __init__(self, show_hud=True, show_landmarks=True, show_scanner=True):
        self.show_hud = show_hud
        self.show_landmarks = show_landmarks
        self.show_scanner = show_scanner
        self.start_time = time.time()
        self.frame_count = 0

    def draw_corner_brackets(self, img, box, color, length=20, thickness=2):
        """Draws futuristic corner targeting brackets instead of a plain rectangle."""
        x, y, w, h = [int(v) for v in box]
        l = min(length, w // 4, h // 4)
        
        # Top-Left
        cv2.line(img, (x, y), (x + l, y), color, thickness)
        cv2.line(img, (x, y), (x, y + l), color, thickness)
        # Top-Right
        cv2.line(img, (x + w, y), (x + w - l, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + l), color, thickness)
        # Bottom-Left
        cv2.line(img, (x, y + h), (x + l, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - l), color, thickness)
        # Bottom-Right
        cv2.line(img, (x + w, y + h), (x + w - l, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - l), color, thickness)

    def draw_scanner_line(self, overlay, box, color):
        """Draws an animated glowing horizontal scanning beam across the face box."""
        x, y, w, h = [int(v) for v in box]
        if h <= 0 or w <= 0:
            return
        t = (time.time() - self.start_time) * 2.5
        phase = (math.sin(t) + 1.0) / 2.0  # 0.0 to 1.0
        scan_y = int(y + phase * h)
        
        cv2.line(overlay, (x, scan_y), (x + w, scan_y), color, 2)
        # Subtle glow gradient
        glow_height = 4
        for i in range(1, glow_height):
            if y <= scan_y - i <= y + h:
                cv2.line(overlay, (x, scan_y - i), (x + w, scan_y - i), color, 1)
            if y <= scan_y + i <= y + h:
                cv2.line(overlay, (x, scan_y + i), (x + w, scan_y + i), color, 1)

    def draw_glass_card(self, img, x, y, w, h, bg_color=(20, 22, 28), alpha=0.75, border_color=None):
        """Draws a semi-transparent glassmorphism card with rounded or beveled border."""
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
        if x2 <= x1 or y2 <= y1:
            return
        
        sub_img = img[y1:y2, x1:x2]
        colored_rect = np.zeros_like(sub_img)
        colored_rect[:] = bg_color
        cv2.addWeighted(colored_rect, alpha, sub_img, 1.0 - alpha, 0, sub_img)
        
        if border_color:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)

    def render_hud(self, frame, tracked_faces, fps=0.0):
        """
        Renders complete biometric HUD onto the frame.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        output = frame.copy()
        overlay = frame.copy()

        # 1. Top HUD Status Bar
        self.draw_glass_card(output, 15, 12, w - 30, 36, bg_color=self.COLOR_DARK, alpha=0.8, border_color=(60, 60, 70))
        cv2.putText(output, "BIOMETRIC AI HUD", (28, 36), cv2.FONT_HERSHEY_DUPLEX, 0.55, self.COLOR_CYAN, 1, cv2.LINE_AA)
        
        status_text = f"TARGETS: {len(tracked_faces)} | FPS: {fps:.1f} | SENSORS: ONLINE"
        cv2.putText(output, status_text, (200, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # 2. Render each tracked face
        for face in tracked_faces:
            box = face.box
            x, y, bw, bh = [int(v) for v in box]
            is_identified = face.name != "Unknown"
            primary_color = self.COLOR_CYAN if is_identified else self.COLOR_AMBER
            
            # Draw Corner Brackets
            self.draw_corner_brackets(output, box, primary_color, length=24, thickness=2)
            
            # Draw animated scanner
            if self.show_scanner:
                self.draw_scanner_line(overlay, box, primary_color)

            # Draw Facial Landmarks
            if self.show_landmarks and face.landmarks:
                for lx, ly in face.landmarks:
                    cv2.circle(output, (int(lx), int(ly)), 3, (0, 255, 200), -1)
                    cv2.circle(output, (int(lx), int(ly)), 5, primary_color, 1)

            # Biometric Info Card positioning
            card_w = 210
            card_h = 85
            card_x = x + bw + 10
            if card_x + card_w > w - 10:
                card_x = max(10, x - card_w - 10)
            card_y = max(55, min(y, h - card_h - 20))

            # Draw HUD Info Card
            self.draw_glass_card(output, card_x, card_y, card_w, card_h, bg_color=self.COLOR_DARK, alpha=0.82, border_color=primary_color)
            
            # Connector Line from Face Box to Card
            conn_start_x = x + bw if card_x > x else x
            conn_start_y = y + 15
            conn_end_x = card_x if card_x > x else card_x + card_w
            conn_end_y = card_y + 15
            cv2.line(output, (conn_start_x, conn_start_y), (conn_end_x, conn_end_y), primary_color, 1)

            # Card Header: Name / ID
            if is_identified:
                name_title = f"{face.name.upper()} ({face.name_conf:.0f}%)"
                badge_color = self.COLOR_CYAN
            else:
                name_title = f"SUBJECT #{face.track_id:02d} (UNKNOWN)"
                badge_color = self.COLOR_AMBER

            cv2.putText(output, name_title, (card_x + 10, card_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, badge_color, 1, cv2.LINE_AA)
            cv2.line(output, (card_x + 10, card_y + 26), (card_x + card_w - 10, card_y + 26), (60, 60, 70), 1)

            # Demographics Line: Age & Gender
            demo_text = f"AGE: {int(round(face.age))} yrs  |  {face.gender}"
            cv2.putText(output, demo_text, (card_x + 10, card_y + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_WHITE, 1, cv2.LINE_AA)

            # Emotion Line
            emotion_text = f"MOOD: {face.emotion.upper()}"
            cv2.putText(output, emotion_text, (card_x + 10, card_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 240, 200), 1, cv2.LINE_AA)

        # Blend scanner overlay
        if self.show_scanner:
            cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)

        # 3. Bottom Control Bar
        self.draw_glass_card(output, 15, h - 35, w - 30, 25, bg_color=self.COLOR_DARK, alpha=0.75)
        controls_text = "[R] Enroll Face  |  [S] Toggle Scanner  |  [L] Toggle Landmarks  |  [Q] Exit"
        cv2.putText(output, controls_text, (25, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 190), 1, cv2.LINE_AA)

        return output
