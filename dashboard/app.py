import streamlit as st

from pages.home import show_home
from pages.inference import show_inference
from pages.health import show_health
from pages.analytics import show_analytics
from pages.explainability import show_xai
from pages.system_info import show_system

from styles import load_css

st.set_page_config(
    page_title="AI Cattle Health Monitoring",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

st.sidebar.title("🐄 AI Cattle Health")
st.sidebar.caption("Multi-Modal Monitoring")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📹 Live Inference",
        "🐄 Cow Health",
        "📊 Analytics",
        "🧠 Explainable AI",
        "⚙ System Information"
    ]
)

if page == "🏠 Home":
    show_home()

elif page == "📹 Live Inference":
    show_inference()

elif page == "🐄 Cow Health":
    show_health()

elif page == "📊 Analytics":
    show_analytics()

elif page == "🧠 Explainable AI":
    show_xai()

elif page == "⚙ System Information":
    show_system()