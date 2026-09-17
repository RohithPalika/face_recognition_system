import cv2
import argparse
import os
import sys
from datetime import datetime
from core.pipeline import BiometricPipeline

def main():
    parser = argparse.ArgumentParser(description="Futuristic Biometric AI Face Recognition & HUD System")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--score-thresh", type=float, default=0.6, help="Face detection score threshold")
    parser.add_argument("--no-hud", action="store_true", help="Disable Sci-Fi HUD overlay")
    args = parser.parse_args()

    print("=" * 60)
    print("   BIOMETRIC AI FACE RECOGNITION & DEMOGRAPHICS SYSTEM")
    print("=" * 60)
    print("Initializing deep learning neural networks...")
    
    pipeline = BiometricPipeline(score_threshold=args.score_thresh)
    if args.no_hud:
        pipeline.visualizer.show_hud = False

    # 1. Image Mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Could not decode image: {args.image}")
            sys.exit(1)

        print(f"Analyzing image: {args.image} ...")
        annotated_frame, tracked_faces, raw_data = pipeline.process_frame(img, log_attendance=False)
        
        print(f"\nDetection Results ({len(tracked_faces)} face(s) detected):")
        for i, face in enumerate(tracked_faces, 1):
            status = f"IDENTIFIED as '{face.name}' ({face.name_conf:.1f}%)" if face.name != "Unknown" else "UNREGISTERED"
            print(f"  [{i}] Status: {status}")
            print(f"      Demographics: Age ~{face.age:.1f} yrs | Gender: {face.gender}")
            print(f"      Facial Expression: {face.emotion}")

        out_name = f"analyzed_{os.path.basename(args.image)}"
        cv2.imwrite(out_name, annotated_frame)
        print(f"\nAnnotated image saved to: {out_name}")

        cv2.imshow("Biometric AI Analysis", annotated_frame)
        print("\nPress any key in the window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 2. Video / Webcam Mode
    source = args.video if args.video else args.camera
    print(f"\nConnecting to video stream (Source: {source})...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        sys.exit(1)

    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Biometric AI Face Recognition HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nSystem Online! Keyboard Controls:")
    print("  [R] - Enroll / Register currently detected face")
    print("  [S] - Toggle Scanner Line animation")
    print("  [L] - Toggle Landmark tracking points")
    print("  [C] - Capture snapshot screenshot")
    print("  [Q] or [ESC] - Exit application\n")

    latest_raw_data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or camera disconnected.")
                break

            annotated_frame, tracked_faces, latest_raw_data = pipeline.process_frame(frame, log_attendance=True)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s') or key == ord('S'):
                pipeline.visualizer.show_scanner = not pipeline.visualizer.show_scanner
                print(f"Scanner animation: {'ON' if pipeline.visualizer.show_scanner else 'OFF'}")
            elif key == ord('l') or key == ord('L'):
                pipeline.visualizer.show_landmarks = not pipeline.visualizer.show_landmarks
                print(f"Landmark tracking points: {'ON' if pipeline.visualizer.show_landmarks else 'OFF'}")
            elif key == ord('c') or key == ord('C'):
                os.makedirs("captures", exist_ok=True)
                filename = f"captures/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved snapshot to: {filename}")
            elif key == ord('r') or key == ord('R'):
                if latest_raw_data:
                    print("\n" + "=" * 40)
                    name = input("Enter person name for enrollment: ").strip()
                    if name:
                        success, msg = pipeline.enroll_from_image(name, frame)
                        print(msg)
                    print("=" * 40 + "\n")
                else:
                    print("No face detected in frame to enroll.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Exited cleanly.")

if __name__ == "__main__":
    main()
