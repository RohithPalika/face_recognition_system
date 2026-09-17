import cv2
import os
import sys
import time
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime

from core.pipeline import BiometricPipeline

def get_name_from_popup(parent=None):
    """Opens a top-level graphical input dialog to enter person's name for enrollment."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    name = simpledialog.askstring("Biometric AI Enrollment", "Enter name of the person to register:", parent=root)
    root.destroy()
    return name.strip() if name else None

def show_popup_message(title, message):
    """Shows a quick popup message."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    messagebox.showinfo(title, message, parent=root)
    root.destroy()

def main():
    print("=" * 65)
    print("      👁️ BIOMETRIC AI - LIVE CAMERA APPLICATION")
    print("=" * 65)
    print("Initializing Neural Networks (YuNet, SFace, FER+, Demographics)...")
    
    pipeline = BiometricPipeline()
    camera_index = 0
    
    print(f"Connecting to live camera feed (Device Index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not access camera at index {camera_index}.")
        print("Please ensure your webcam is plugged in and permissions are allowed.")
        sys.exit(1)

    # Set camera resolution to 1280x720 HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Biometric AI - Live Camera HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n✅ Camera Online! Controls:")
    print("   [R] - Register / Enroll your face into the database")
    print("   [S] - Toggle Sci-Fi Scanner beam")
    print("   [L] - Toggle 5-point Facial Landmarks")
    print("   [H] - Toggle Biometric HUD Overlay")
    print("   [C] - Capture snapshot image")
    print("   [Q] or [ESC] - Quit Application\n")

    latest_raw_data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera feed lost. Exiting...")
                break

            # Mirror the camera horizontally for a natural webcam mirror view
            frame = cv2.flip(frame, 1)

            # Process live frame through AI pipeline
            annotated_frame, tracked_faces, latest_raw_data = pipeline.process_frame(frame, log_attendance=True)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord('q') or key == 27:
                break
            
            # Register Face (Popup Dialog)
            elif key == ord('r') or key == ord('R'):
                if latest_raw_data:
                    # Pause and get name from popup
                    name = get_name_from_popup()
                    if name:
                        success, msg = pipeline.enroll_from_image(name, frame)
                        print(f"[Enrollment] {msg}")
                        show_popup_message("Enrollment Status", f"✅ {msg}")
                else:
                    print("[Enrollment] No face detected to register. Please look directly at the camera.")
                    show_popup_message("Enrollment Error", "❌ No face detected in camera view. Look directly at the camera and try again.")

            # Toggle Scanner
            elif key == ord('s') or key == ord('S'):
                pipeline.visualizer.show_scanner = not pipeline.visualizer.show_scanner
                print(f"[HUD] Scanner Line: {'ON' if pipeline.visualizer.show_scanner else 'OFF'}")

            # Toggle Landmarks
            elif key == ord('l') or key == ord('L'):
                pipeline.visualizer.show_landmarks = not pipeline.visualizer.show_landmarks
                print(f"[HUD] Landmark Tracking: {'ON' if pipeline.visualizer.show_landmarks else 'OFF'}")

            # Toggle HUD
            elif key == ord('h') or key == ord('H'):
                pipeline.visualizer.show_hud = not pipeline.visualizer.show_hud
                print(f"[HUD] Overlay Display: {'ON' if pipeline.visualizer.show_hud else 'OFF'}")

            # Capture Snapshot
            elif key == ord('c') or key == ord('C'):
                os.makedirs("captures", exist_ok=True)
                snapshot_file = f"captures/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snapshot_file, annotated_frame)
                print(f"[Snapshot] Saved to: {snapshot_file}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera disconnected and resources released cleanly.")

if __name__ == "__main__":
    main()
