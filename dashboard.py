import streamlit as st
import os
import tempfile
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_DIR, exist_ok=True)
# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="AI Cattle Health Monitoring",
    page_icon="🐄",
    layout="wide"
)

# ============================================
# Title
# ============================================

st.title("🐄 AI-Powered Multi-Modal Cattle Health Monitoring System")

st.markdown(
"""
Early Disease Detection using

- Computer Vision
- Sensor AI
- Decision Fusion
- Explainable AI (XAI)
"""
)

st.divider()

# ============================================
# Sidebar
# ============================================

st.sidebar.title("Dashboard")

page = st.sidebar.radio(

    "Navigation",

    [

        "Inference",

        "System Information"

    ]

)

# ============================================
# Main Page
# ============================================

if page == "Inference":

    st.header("Real-Time Health Inference")

    col1, col2 = st.columns(2)

    # -----------------------------------

    # Upload Video

    # -----------------------------------

    with col1:

        st.subheader("Upload Cow Video")

        uploaded_video = st.file_uploader(

            "Choose MP4 Video",

            type=["mp4"]

        )

        if uploaded_video:

            st.success("Video Uploaded Successfully")

            st.video(uploaded_video)

    # -----------------------------------

    # Upload Sensor CSV

    # -----------------------------------

    with col2:

        st.subheader("Upload Sensor CSV")

        uploaded_csv = st.file_uploader(

            "Choose Sensor CSV",

            type=["csv"]

        )

        if uploaded_csv:

            st.success("CSV Uploaded Successfully")

    st.divider()

    run_button = st.button(

        "Run AI Pipeline",

        use_container_width=True

    )

    if run_button:

        if uploaded_video is None:

            st.error("Please upload a video.")

        elif uploaded_csv is None:

            st.error("Please upload sensor CSV.")

        else:

            st.success("Everything Ready!")

            progress = st.progress(0)

            status = st.empty()

            status.info("Saving uploaded video...")

            progress.progress(10)

            temp_video_path = os.path.join(
                TEMP_DIR,
                uploaded_video.name
            )

            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            status.success("Video Saved")

            progress.progress(30)

        status.info("Running Vision Pipeline...")
        vision_script = os.path.join(
            BASE_DIR,
            "realtime_inference.py"
        )

        try:

            result = subprocess.run(

                [
                    "python3",
                    vision_script,
                    temp_video_path
                ],

                capture_output=True,
                text=True
            )

            progress.progress(90)

            if result.returncode == 0:

                status.success("Vision Pipeline Completed")

                st.code(result.stdout)

            else:

                status.error("Vision Pipeline Failed")

                st.code(result.stderr)

        except Exception as e:

            st.error(e)

# ============================================

elif page == "System Information":

    st.header("System Information")

    st.write("Model : XGBoost")

    st.write("Vision : YOLO + ByteTrack + Geometry")

    st.write("Decision : Rule-based Fusion")

    st.write("Explainability : XAI")