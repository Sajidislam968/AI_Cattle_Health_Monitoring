import streamlit as st


def show_home():

    st.title("🐄 AI Powered Multi Modal Cattle Health Monitoring")

    st.markdown("""
Early Disease Detection using

- YOLO11
- Behavior Analysis
- Synthetic Collar Sensors
- XGBoost
- Decision Fusion
- Explainable AI
""")

    st.divider()

    c1,c2,c3,c4,c5,c6=st.columns(6)

    c1.metric("Healthy",18)
    c2.metric("At Risk",3)
    c3.metric("Diseased",1)
    c4.metric("Avg Confidence","89%")
    c5.metric("Videos","25")
    c6.metric("System","Online")

    st.divider()

    st.subheader("System Status")

    st.success("Vision Pipeline Ready")

    st.success("Sensor Generator Ready")

    st.success("Decision Fusion Ready")

    st.success("Recommendation Engine Ready")

    st.success("Explainable AI Ready")