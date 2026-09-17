import os
import csv
from datetime import datetime
import pandas as pd

class AttendanceLogger:
    """Manages biometric detection and attendance logs with CSV export and pandas querying."""

    def __init__(self, log_dir="logs", log_file="attendance_log.csv"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        self.headers = ["Timestamp", "Date", "Time", "Person_Name", "Confidence", "Age", "Gender", "Emotion", "Status"]
        self.cooldown = {}  # name -> last_logged_timestamp to avoid spamming the log every frame
        
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_detection(self, name, confidence, age, gender, emotion, cooldown_seconds=60):
        """Logs a recognized person if cooldown duration has passed."""
        now = datetime.now()
        now_ts = now.timestamp()
        
        # Don't log Unknown rapidly unless cooldown passed
        key = name
        if key in self.cooldown:
            if now_ts - self.cooldown[key] < cooldown_seconds:
                return False

        self.cooldown[key] = now_ts
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        status = "VERIFIED" if name != "Unknown" else "UNREGISTERED"

        row = [
            now.isoformat(),
            date_str,
            time_str,
            name,
            f"{confidence:.1f}%",
            int(round(age)),
            gender,
            emotion,
            status
        ]

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return True

    def get_logs_df(self):
        """Returns log history as a pandas DataFrame."""
        if not os.path.exists(self.log_path):
            return pd.DataFrame(columns=self.headers)
        try:
            return pd.read_csv(self.log_path)
        except Exception:
            return pd.DataFrame(columns=self.headers)

    def clear_logs(self):
        """Clears existing logs."""
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
