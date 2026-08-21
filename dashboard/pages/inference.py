import streamlit as st
import requests


def show_inference():

    st.title("📹 Live Inference")

    left,right=st.columns(2)

    with left:

        st.subheader("Upload Video")

        uploaded_file = st.file_uploader(
            "Upload MP4",
            type=["mp4"]
        )

    with right:

        st.subheader("ESP32 Collar Sensors")

        st.info("No physical ESP32 connected.")

        st.write("Synthetic Sensor Generator will simulate live collar data.")

    st.divider()

    if st.button(
        "Run Complete Pipeline",
        use_container_width=True
    ):
        if uploaded_file is None:
            st.error("❌ Please upload a video first")
        else:
            st.info("⏳ Processing video... This may take a few minutes")
            
            try:
                files = {"video": (uploaded_file.name, uploaded_file.read(), "video/mp4")}
                # Increase timeout to 1 hour (3600s) for long video processing
                response = requests.post("http://localhost:8000/run", files=files, timeout=3600)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Pipeline completed successfully!")
                    
                    st.subheader("📊 Results")
                    progress_container = st.container()
                    with progress_container:
                        for log in result.get("logs", []):
                            st.write(f"**{log['step']}**: {log['time']}s")
                    
                    if result.get("result_csv"):
                        st.success(f"✅ Results saved to: {result['result_csv']}")
                else:
                    st.error(f"❌ Error: {response.text}")
            except requests.exceptions.Timeout:
                st.error("❌ Video processing timed out. The video might be too long. Try a shorter video.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API server. Make sure FastAPI is running on http://localhost:8000")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")