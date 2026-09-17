import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime

from core.pipeline import BiometricPipeline

# Page configuration
st.set_page_config(
    page_title="Biometric AI - Face Recognition & Analytics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Futuristic Glassmorphic Dark Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;600;700&display=swap');

    .main {
        background-color: #0b0e14;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(14, 23, 42, 0.95) 0%, rgba(6, 9, 15, 1) 90%);
    }

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 2.2rem;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: 1px;
    }

    .sub-title {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-verified { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
    .badge-unregistered { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
</style>
""", unsafe_allow_html=True)

# Initialize Pipeline in Streamlit Session Cache
@st.cache_resource
def get_pipeline():
    return BiometricPipeline()

pipeline = get_pipeline()

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="main-title">BIOMETRIC AI</div>', unsafe_allow_html=True)
    st.caption("Deep Learning Face Recognition & Demographics")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation",
        ["📸 Photo & Video Analysis", "👤 Face Database Manager", "📊 Attendance & Security Logs", "ℹ️ System Info"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("⚙️ Detection Settings")
    score_thresh = st.slider("Confidence Threshold", 0.3, 0.9, 0.6, 0.05)
    show_landmarks = st.checkbox("Show Face Landmarks", value=True)
    show_scanner = st.checkbox("Show Scanner Line", value=True)
    
    pipeline.detector.score_threshold = score_thresh
    pipeline.visualizer.show_landmarks = show_landmarks
    pipeline.visualizer.show_scanner = show_scanner

# -------------------------------------------------------------
# TAB 1: Photo & Video Analysis
# -------------------------------------------------------------
if app_mode == "📸 Photo & Video Analysis":
    st.markdown('<div class="main-title">Real-Time Biometric Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detect faces, recognize enrolled identities, estimate continuous age, gender, and facial expressions.</div>', unsafe_allow_html=True)

    input_source = st.radio("Select Input Source", ["Upload Image", "Webcam Snapshot"], horizontal=True)
    
    img_bgr = None
    
    if input_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload an Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            # Preset sample images
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            sample_choice = None
            if col_s1.button("Sample: Girl 1"): sample_choice = "girl1.jpg"
            if col_s2.button("Sample: Man 1"): sample_choice = "man1.jpg"
            if col_s3.button("Sample: Kid 1"): sample_choice = "kid1.jpg"
            if col_s4.button("Sample: Woman 1"): sample_choice = "woman1.jpg"
            
            if sample_choice and os.path.exists(sample_choice):
                img_bgr = cv2.imread(sample_choice)
                
    elif input_source == "Webcam Snapshot":
        camera_img = st.camera_input("Take a photo with your webcam")
        if camera_img is not None:
            file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is not None:
        with st.spinner("Processing biometric deep neural networks..."):
            annotated_frame, tracked_faces, raw_data = pipeline.process_frame(img_bgr, log_attendance=True)

        col_img, col_metrics = st.columns([1.3, 1])

        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("HUD Visualizer Output")
            # Convert BGR to RGB for Streamlit
            img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_metrics:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(f"Detections Summary ({len(tracked_faces)} Face(s))")
            
            if not tracked_faces:
                st.warning("No faces detected in the image.")
            else:
                for i, face in enumerate(tracked_faces, 1):
                    is_known = face.name != "Unknown"
                    badge_class = "badge-verified" if is_known else "badge-unregistered"
                    status_label = f"VERIFIED: {face.name}" if is_known else "UNREGISTERED"

                    st.markdown(f"""
                    <div style="border-left: 3px solid {'#00f2fe' if is_known else '#f59e0b'}; padding-left: 10px; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #fff;">Subject #{face.track_id}</h4>
                            <span class="metric-badge {badge_class}">{status_label}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    c1.metric("Identity Match", face.name, f"{face.name_conf:.1f}%" if is_known else "N/A")
                    c2.metric("Estimated Age", f"{int(round(face.age))} yrs", f"Exact: {face.age:.1f}")

                    c3, c4 = st.columns(2)
                    c3.metric("Gender", face.gender)
                    c4.metric("Facial Expression", f"{face.emotion}")

                    # Detailed Emotion Breakdown if available
                    if i <= len(raw_data) and "emo_full" in raw_data[i-1]:
                        emo_scores = raw_data[i-1]["emo_full"]["scores"]
                        st.caption("Emotion Probability Distribution")
                        df_emo = pd.DataFrame(list(emo_scores.items()), columns=["Emotion", "Probability"])
                        df_emo["Percentage"] = df_emo["Probability"] * 100
                        st.bar_chart(df_emo.set_index("Emotion")["Percentage"])

                    st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: Face Database Manager
# -------------------------------------------------------------
elif app_mode == "👤 Face Database Manager":
    st.markdown('<div class="main-title">Face Database & Enrollment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Register new known identities to enable automatic face recognition.</div>', unsafe_allow_html=True)

    col_enroll, col_gallery = st.columns([1, 1.2])

    with col_enroll:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("➕ Enroll New Person")
        
        person_name = st.text_input("Person Full Name", placeholder="e.g. Alex Johnson")
        enroll_img_file = st.file_uploader("Upload Person Photo", type=["jpg", "png", "jpeg"], key="enroll_file")

        if st.button("🚀 Enroll into Database", type="primary"):
            if not person_name.strip():
                st.error("Please enter a valid name.")
            elif enroll_img_file is None:
                st.error("Please upload an image of the person.")
            else:
                file_bytes = np.asarray(bytearray(enroll_img_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                success, msg = pipeline.enroll_from_image(person_name.strip(), img)
                if success:
                    st.success(f"✅ {msg}")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_gallery:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📁 Enrolled Identities")
        
        enrolled = pipeline.recognizer.get_enrolled_names()
        if not enrolled:
            st.info("No identities enrolled yet. Enroll someone using the form on the left!")
        else:
            st.write(f"Total Enrolled Profiles: **{len(enrolled)}**")
            for name in enrolled:
                g_col1, g_col2, g_col3 = st.columns([1, 2, 1])
                photo_path = os.path.join(pipeline.recognizer.db_dir, f"{name}.jpg")
                with g_col1:
                    if os.path.exists(photo_path):
                        thumb = Image.open(photo_path)
                        st.image(thumb, width=70)
                    else:
                        st.write("👤")
                with g_col2:
                    st.markdown(f"**{name}**")
                    st.caption("Biometric embedding registered")
                with g_col3:
                    if st.button("🗑️ Delete", key=f"del_{name}"):
                        pipeline.recognizer.delete_person(name)
                        st.success(f"Deleted {name}")
                        st.rerun()
                st.markdown("---")
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: Attendance & Security Logs
# -------------------------------------------------------------
elif app_mode == "📊 Attendance & Security Logs":
    st.markdown('<div class="main-title">Attendance & Security Audit Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time timestamped event logs of recognized persons and demographic stats.</div>', unsafe_allow_html=True)

    df_logs = pipeline.logger.get_logs_df()

    if df_logs.empty:
        st.info("No attendance records logged yet. Process images or webcam feeds to generate logs!")
    else:
        # Quick metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Detections", len(df_logs))
        verified_count = len(df_logs[df_logs["Status"] == "VERIFIED"]) if "Status" in df_logs else 0
        m2.metric("Verified Faces", verified_count)
        unique_people = df_logs["Person_Name"].nunique() if "Person_Name" in df_logs else 0
        m3.metric("Unique Individuals", unique_people)
        avg_age = round(df_logs["Age"].mean(), 1) if "Age" in df_logs and len(df_logs) > 0 else 0
        m4.metric("Avg Estimated Age", f"{avg_age} yrs")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📋 Detection Events Log")
        
        # Search & Filter
        search_query = st.text_input("🔍 Search Person Name", "")
        if search_query:
            filtered_df = df_logs[df_logs["Person_Name"].str.contains(search_query, case=False, na=False)]
        else:
            filtered_df = df_logs

        st.dataframe(filtered_df, use_container_width=True)

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export CSV Log",
                csv_data,
                "attendance_log.csv",
                "text/csv",
                key='download-csv'
            )
        with col_clr:
            if st.button("🗑️ Clear All Logs"):
                pipeline.logger.clear_logs()
                st.success("Audit logs cleared.")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Analytics Charts
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📈 Demographics & Expression Analytics")
        ch1, ch2 = st.columns(2)
        with ch1:
            if "Gender" in df_logs:
                st.caption("Gender Distribution")
                st.bar_chart(df_logs["Gender"].value_counts())
        with ch2:
            if "Emotion" in df_logs:
                st.caption("Facial Expression Breakdown")
                st.bar_chart(df_logs["Emotion"].value_counts())
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 4: System Info
# -------------------------------------------------------------
elif app_mode == "ℹ️ System Info":
    st.markdown('<div class="main-title">System Architecture & Capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">State-of-the-art Deep Learning Computer Vision Pipeline.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>🧠 Deep Learning Models Under the Hood</h3>
        <ul>
            <li><b>YuNet Face Detector (ONNX)</b>: High-speed, robust face detection with 5-point facial landmark alignment (eyes, nose, mouth).</li>
            <li><b>SFace Face Recognition (ONNX)</b>: ArcFace 128D deep feature embeddings for high-accuracy identity verification and cosine matching.</li>
            <li><b>Continuous Age & Gender Estimator</b>: Softmax probability expected value regression for continuous single-year age predictions without discrete jumping.</li>
            <li><b>FER+ Emotion Recognition (ONNX)</b>: 8-class facial expression analysis (Happy, Neutral, Surprise, Sad, Angry, Fear, Disgust, Contempt).</li>
            <li><b>Centroid & IOU Temporal Tracker</b>: Exponential Moving Average (EMA) smoothing for rock-solid bounding boxes and flicker-free predictions.</li>
            <li><b>Futuristic Biometric HUD</b>: Sci-Fi glowing corner targeting, animated scanner beams, and glassmorphic telemetry cards.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
