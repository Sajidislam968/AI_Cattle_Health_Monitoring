import streamlit as st


def load_css():

    st.markdown("""
    <style>

    [data-testid="stAppViewContainer"] {
        background: #f6f9fc;
        color: #17324d;
    }

    [data-testid="stHeader"] {
        background: #f6f9fc;
    }

    [data-testid="stSidebar"] {
        background: #edf3f7;
        border-right: 1px solid #dbe6ee;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe6ee;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 16px rgba(23, 50, 77, 0.07);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span {
        color: #526579 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] span {
        color: #17324d !important;
    }

    [data-testid="stMetricDelta"] {
        color: #2266a8 !important;
    }

    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
        color: #17324d;
    }

    [data-testid="stCaptionContainer"] {
        color: #526579;
    }

    .stButton > button,
    [data-testid="stDownloadButton"] button {
        color: #ffffff;
        background: #2266a8;
        border: 1px solid #2266a8;
        border-radius: 8px;
    }

    .stButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {
        color: #ffffff;
        background: #174d80;
        border-color: #174d80;
    }

    [data-testid="stAlert"] {
        color: #17324d;
    }

    .block-container {
        padding-top: 2rem;
    }

    </style>
    """, unsafe_allow_html=True)