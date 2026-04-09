import streamlit as st
import os
import time
import tempfile
import re
from html import escape
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

# Set up environment variables
load_dotenv()

# Get API keys with defaults to avoid NoneType errors
google_api_key = os.getenv("GOOGLE_API_KEY", "")
groq_api_key = os.getenv("GROQ_API_KEY", "")

if google_api_key:
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Streamlit UI setup
st.set_page_config(page_title="LawGPT", layout="wide")
import os

# Main hero section
hero_col1 = st.container()

with hero_col1:
    st.markdown(
        """
        <div class="hero-content-wrapper">
            <div class="hero-kicker">Indian law assistant</div>
            <h1 class="hero-title">LawGPT Legal Assistant</h1>
            <p class="hero-copy">
                Ask focused questions about Indian law, legal rights, court procedure, contracts,
                notices, or uploaded legal documents. Off-topic questions are intentionally declined.
            </p>
            <div class="hero-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

    :root {
        --law-ink: #1b2430;
        --law-navy: #1a2f4f;
        --law-gold: #8b7a4d;
        --law-gold-soft: #b8aa7e;
        --law-paper: #e6e0d2;
        --law-panel: rgba(244, 240, 230, 0.9);
        --law-border: #b9ad8e;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 12%, rgba(139, 122, 77, 0.18), transparent 34%),
            radial-gradient(circle at 92% 88%, rgba(26, 47, 79, 0.2), transparent 38%),
            linear-gradient(145deg, #d8d1bf 0%, #cdc4af 46%, #c2b79b 100%);
        color: var(--law-ink);
        font-family: 'Source Sans 3', serif;
    }

    [data-testid="stToolbar"],
    [data-testid="stAppToolbar"],
    [data-testid="stHeaderActionElements"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    #MainMenu,
    .stDeployButton,
    footer,
    #stDecoration,
    button[title="View fullscreen"] {
        visibility: hidden;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .hero-content-wrapper {
        padding: 1.35rem 1.5rem 1.35rem 0;
        animation: fadeInUp 0.7s ease-out;
    }

    .hero-title {
        margin: 0;
        border-bottom: 2px solid rgba(139, 122, 77, 0.62);
        padding-bottom: 0.35rem;
        font-size: 2.4rem;
        font-family: 'Cinzel', serif;
        color: var(--law-navy);
        letter-spacing: 0.02em;
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        right: -2rem;
        top: -2rem;
        width: 10rem;
        height: 10rem;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(26, 47, 79, 0.12), rgba(26, 47, 79, 0));
        pointer-events: none;
    }

    .hero-kicker {
        display: inline-block;
        margin-bottom: 0.45rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(26, 47, 79, 0.14);
        color: var(--law-navy);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-shell h1 {
        margin: 0;
        border-bottom: none;
        padding-bottom: 0;
        font-size: 2.4rem;
    }

    .hero-copy {
        max-width: 56rem;
        margin: 0.45rem 0 0;
        color: #2a3747;
        font-size: 1.02rem;
        line-height: 1.65;
    }

    .hero-rule {
        width: 7.5rem;
        height: 4px;
        margin-top: 1rem;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--law-gold), rgba(139, 122, 77, 0.2));
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    .block-container {
        background: var(--law-panel);
        border: 1px solid var(--law-border);
        border-radius: 18px;
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        box-shadow: 0 14px 35px rgba(26, 47, 79, 0.1);
    }

    h1, h2, h3 {
        font-family: 'Cinzel', serif;
        color: var(--law-navy);
        letter-spacing: 0.02em;
    }

    h1 {
        border-bottom: 2px solid rgba(139, 122, 77, 0.62);
        padding-bottom: 0.35rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #e8e2d3;
        border: 1px dashed var(--law-gold);
        border-radius: 14px;
    }

    div.stButton > button:first-child {
        background: linear-gradient(125deg, var(--law-navy), #29446e);
        color: #f4f0e3;
        border: 1px solid #2e4d7b;
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(26, 47, 79, 0.28);
        border-color: #385a8d;
    }

    div.stButton > button:active {
        transform: translateY(0);
    }

    [data-baseweb="select"] > div,
    .stTextInput > div > div,
    .stChatInput > div {
        background: #f6f2e8;
        border-radius: 10px;
        border: 1px solid var(--law-border);
    }

    [data-testid="stChatMessage"] {
        background: rgba(247, 243, 234, 0.9);
        border: 1px solid rgba(185, 173, 142, 0.88);
        border-radius: 12px;
        padding: 0.35rem 0.55rem;
    }

    .section-shell {
        padding: 1rem 1.1rem;
        margin-top: 0.9rem;
        border-radius: 18px;
        background: rgba(242, 236, 223, 0.78);
        border: 1px solid rgba(185, 173, 142, 0.75);
        box-shadow: 0 10px 22px rgba(26, 47, 79, 0.08);
    }

    .section-label {
        display: inline-flex;
        margin-bottom: 0.65rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        background: rgba(139, 122, 77, 0.22);
        color: #4a3f27;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .stMarkdown hr {
        border-top: 1px solid rgba(139, 122, 77, 0.52);
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-18px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes pulseGold {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(139, 122, 77, 0.42);
        }
        50% {
            box-shadow: 0 0 0 8px rgba(139, 122, 77, 0);
        }
    }

    @keyframes shimmerLaw {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }

    @keyframes fadeInRight {
        from {
            opacity: 0;
            transform: translateX(24px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    .hero-shell {
        animation: fadeInUp 0.7s ease-out;
    }

    .hero-kicker {
        animation: slideInLeft 0.6s ease-out 0.15s both;
    }

    .hero-shell h1 {
        animation: slideInLeft 0.6s ease-out 0.25s both;
    }

    .hero-copy {
        animation: slideInLeft 0.6s ease-out 0.35s both;
    }

    .hero-rule {
        animation: slideInLeft 0.6s ease-out 0.45s both;
    }

    .section-shell {
        animation: fadeInUp 0.5s ease-out;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .section-shell:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 32px rgba(26, 47, 79, 0.16) !important;
    }

    [data-testid="stChatMessage"] {
        animation: fadeInUp 0.4s ease-out;
        transition: background 0.3s ease, border-color 0.3s ease;
    }

    div.stButton > button:first-child {
        position: relative;
        animation: fadeInUp 0.5s ease-out;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    div.stButton > button:first-child:active {
        animation: pulseGold 0.6s ease-out;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 35px,
                rgba(139, 122, 77, 0.05) 35px,
                rgba(139, 122, 77, 0.05) 70px
            ),
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 35px,
                rgba(26, 47, 79, 0.03) 35px,
                rgba(26, 47, 79, 0.03) 70px
            );
        pointer-events: none;
        z-index: 0;
    }

    .legal-accent {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--law-gold);
        margin-right: 0.4rem;
        animation: pulseGold 2.5s infinite;
    }

    /* Hide/minimize loading and status containers */
    [data-testid="stStatusContainer"],
    .stStatus {
        display: none !important;
    }

    @media (max-width: 768px) {
        .hero-content-wrapper {
            padding: 1rem 0;
        }

        .hero-title {
            font-size: 1.8rem;
        }

        .hero-copy {
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .hero-kicker {
            font-size: 0.7rem;
        }

        h1, h2, h3 {
            font-family: 'Cinzel', serif;
        }

        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1rem;
            margin: 0.4rem 0;
        }

        .section-shell {
            padding: 0.8rem 0.9rem;
            margin-top: 0.6rem;
        }

        [data-baseweb="select"] > div,
        .stTextInput > div > div,
        .stChatInput > div {
            font-size: 0.95rem;
        }

        .stButton {
            width: 100%;
        }

        [data-testid="stChatMessage"] {
            padding: 0.6rem 0.8rem;
            border-radius: 10px;
        }

        .section-label {
            font-size: 0.65rem;
            padding: 0.25rem 0.6rem;
        }

        [data-testid="stFileUploaderDropzone"] {
            border-radius: 10px;
            padding: 1rem;
        }
    }

    @media (max-width: 480px) {
        .hero-shell {
            padding: 0.8rem 1rem;
            gap: 0.8rem;
        }

        .hero-shell h1 {
            font-size: 1.5rem;
        }

        .hero-copy {
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .hero-icon {
            max-width: 100px;
        }

        .hero-icon svg {
            width: 85px !important;
            height: 100px !important;
        }

        .block-container {
            padding: 0.6rem 0.8rem;
        }

        .section-shell {
            padding: 0.6rem 0.7rem;
            margin-top: 0.4rem;
        }

        [data-testid="stChatMessage"] {
            padding: 0.5rem 0.6rem;
            font-size: 0.9rem;
        }

        [data-baseweb="select"] > div,
        .stChatInput > div {
            font-size: 0.9rem;
            padding: 0.6rem;
        }

        .section-label {
            font-size: 0.6rem;
            padding: 0.2rem 0.5rem;
        }

        div.stButton > button:first-child {
            font-size: 0.9rem;
            padding: 0.7rem 1rem;
        }

        h1 {
            border-bottom: 1.5px solid rgba(139, 122, 77, 0.62);
        }

        .stMarkdown hr {
            margin: 0.5rem 0;
        }
    }

    @media (max-width: 360px) {
        .hero-shell {
            padding: 0.7rem 0.9rem;
        }

        .hero-shell h1 {
            font-size: 1.3rem;
            margin-bottom: 0.3rem;
        }

        .hero-icon {
            display: none;
        }

        .hero-copy {
            font-size: 0.85rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Cormorant Garamond', serif;
    }

    .stApp {
        position: relative;
        background:
            radial-gradient(circle at 18% 18%, rgba(255, 215, 140, 0.20), transparent 22%),
            radial-gradient(circle at 82% 12%, rgba(22, 44, 77, 0.18), transparent 26%),
            radial-gradient(circle at 80% 82%, rgba(139, 122, 77, 0.14), transparent 28%),
            linear-gradient(145deg, #f3ebdc 0%, #e8ddc6 36%, #d9ccb2 72%, #cbb995 100%);
        color: #1b2430;
        overflow-x: hidden;
    }

    .stApp::after {
        content: "⚖️";
        position: fixed;
        top: 6%;
        right: 4%;
        font-size: 6rem;
        opacity: 0.08;
        transform: rotate(-10deg);
        animation: driftSeal 10s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(16, 27, 46, 0.98), rgba(24, 38, 62, 0.95)),
            radial-gradient(circle at top, rgba(191, 159, 86, 0.16), transparent 42%);
        border-right: 1px solid rgba(191, 159, 86, 0.38);
        box-shadow: 14px 0 28px rgba(18, 26, 39, 0.20);
    }

    [data-testid="stSidebar"] * {
        color: #f4eddc;
    }

    .history-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(191, 159, 86, 0.45);
        background: linear-gradient(135deg, rgba(191, 159, 86, 0.22), rgba(255, 255, 255, 0.05));
        letter-spacing: 0.22em;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
    }

    .history-panel-title {
        font-family: 'Cinzel', serif;
        font-size: 1.1rem;
        letter-spacing: 0.06em;
        margin: 0.25rem 0 0.6rem;
        color: #fff1d0;
    }

    .history-card {
        padding: 0.7rem 0.8rem;
        margin-bottom: 0.7rem;
        border-radius: 14px;
        border: 1px solid rgba(191, 159, 86, 0.34);
        background: linear-gradient(180deg, rgba(255, 250, 240, 0.10), rgba(255, 255, 255, 0.04));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .history-meta {
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        color: rgba(244, 237, 220, 0.75);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }

    .history-question {
        color: #fff7e6;
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 0.35rem;
    }

    .history-answer {
        color: rgba(244, 237, 220, 0.88);
        font-size: 0.93rem;
        line-height: 1.35;
    }

    .history-empty {
        padding: 0.85rem;
        border-radius: 12px;
        border: 1px dashed rgba(191, 159, 86, 0.38);
        color: rgba(244, 237, 220, 0.70);
        background: rgba(255, 255, 255, 0.04);
        font-size: 0.95rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Cinzel', serif;
        letter-spacing: 0.03em;
    }

    .hero-content-wrapper {
        position: relative;
        padding: 1.35rem 1.5rem 1.35rem 0;
        animation: fadeInUp 0.7s ease-out;
    }

    .hero-content-wrapper::before {
        content: "LAW CHAMBER";
        display: inline-flex;
        margin-bottom: 0.7rem;
        padding: 0.38rem 0.85rem;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(26, 47, 79, 0.14), rgba(191, 159, 86, 0.18));
        border: 1px solid rgba(139, 122, 77, 0.42);
        color: #1a2f4f;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        box-shadow: 0 10px 20px rgba(26, 47, 79, 0.08);
    }

    .hero-title {
        margin: 0;
        border-bottom: 2px solid rgba(139, 122, 77, 0.72);
        padding-bottom: 0.45rem;
        font-size: 2.75rem;
        font-family: 'Cinzel', serif;
        color: #142136;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.65);
    }

    .hero-copy {
        max-width: 62rem;
        margin: 0.65rem 0 0;
        color: #2a3747;
        font-size: 1.08rem;
        line-height: 1.7;
    }

    .hero-rule {
        width: 10rem;
        height: 4px;
        margin-top: 1rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #142136, #8b7a4d, rgba(191, 159, 86, 0.18));
        background-size: 200% 100%;
        animation: ruleShimmer 6s linear infinite;
    }

    .block-container {
        background: rgba(255, 250, 243, 0.78);
        border: 1px solid rgba(139, 122, 77, 0.32);
        border-radius: 22px;
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        box-shadow:
            0 18px 46px rgba(18, 30, 48, 0.10),
            inset 0 1px 0 rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(8px);
    }

    .section-shell {
        position: relative;
        overflow: hidden;
        padding: 1rem 1.15rem;
        margin-top: 0.9rem;
        border-radius: 22px;
        background: linear-gradient(180deg, rgba(255, 251, 245, 0.92), rgba(245, 237, 222, 0.82));
        border: 1px solid rgba(139, 122, 77, 0.28);
        box-shadow: 0 14px 30px rgba(26, 47, 79, 0.10);
    }

    .section-shell::after {
        content: "";
        position: absolute;
        inset: auto -20% -35% auto;
        width: 12rem;
        height: 12rem;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(191, 159, 86, 0.20), transparent 60%);
        animation: driftOrb 12s ease-in-out infinite;
        pointer-events: none;
    }

    .section-label {
        display: inline-flex;
        margin-bottom: 0.65rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(139, 122, 77, 0.22), rgba(26, 47, 79, 0.10));
        color: #43381f;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    [data-testid="stFileUploaderDropzone"],
    [data-baseweb="select"] > div,
    .stTextInput > div > div,
    .stChatInput > div {
        background: rgba(255, 250, 242, 0.95);
        border: 1px solid rgba(139, 122, 77, 0.36);
        border-radius: 14px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }

    [data-testid="stFileUploaderDropzone"] {
        border-style: dashed;
        border-color: rgba(139, 122, 77, 0.55);
    }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #162c4d, #2f507e, #8b7a4d);
        background-size: 220% 220%;
        color: #fff7e8;
        border: 1px solid rgba(191, 159, 86, 0.55);
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.04em;
        box-shadow: 0 12px 24px rgba(18, 30, 48, 0.18);
        transition: transform 0.22s ease, box-shadow 0.22s ease, background-position 0.6s ease;
        animation: fadeInUp 0.5s ease-out;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 30px rgba(18, 30, 48, 0.24);
        background-position: 100% 0;
    }

    div.stButton > button:first-child:active {
        animation: pulseGold 0.6s ease-out;
        transform: translateY(0);
    }

    [data-testid="stChatMessage"] {
        background: linear-gradient(180deg, rgba(255, 249, 239, 0.95), rgba(244, 237, 223, 0.92));
        border: 1px solid rgba(139, 122, 77, 0.26);
        border-radius: 16px;
        padding: 0.55rem 0.7rem;
        box-shadow: 0 8px 20px rgba(26, 47, 79, 0.08);
        animation: fadeInUp 0.4s ease-out;
    }

    [data-testid="stChatMessage"] p {
        color: #1b2430;
    }

    [data-testid="stChatMessage"]:nth-child(even) {
        border-left: 4px solid rgba(26, 47, 79, 0.62);
    }

    [data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 4px solid rgba(139, 122, 77, 0.60);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseGold {
        0%, 100% { box-shadow: 0 0 0 0 rgba(139, 122, 77, 0.42); }
        50% { box-shadow: 0 0 0 8px rgba(139, 122, 77, 0); }
    }

    @keyframes driftSeal {
        0%, 100% { transform: translateY(0) rotate(-10deg); }
        50% { transform: translateY(10px) rotate(-7deg); }
    }

    @keyframes driftOrb {
        0%, 100% { transform: translate(0, 0); opacity: 0.55; }
        50% { transform: translate(-14px, -8px); opacity: 0.82; }
    }

    @keyframes ruleShimmer {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    @media (max-width: 768px) {
        .hero-title { font-size: 2.05rem; }
        .hero-copy { font-size: 0.98rem; line-height: 1.55; }
        .section-shell { padding: 0.85rem 0.95rem; }
        .block-container { padding-top: 0.9rem; padding-bottom: 1rem; }
        [data-testid="stChatMessage"] { padding: 0.45rem 0.55rem; }
    }

    @media (max-width: 480px) {
        .hero-title { font-size: 1.65rem; }
        .hero-content-wrapper::before { font-size: 0.66rem; letter-spacing: 0.14em; }
        .history-card { padding: 0.6rem 0.65rem; }
    }
    </style>
""", unsafe_allow_html=True)

# Check for required API keys and vector store
if not groq_api_key:
    st.error("⚠️ GROQ API key is missing! Please add your GROQ_API_KEY to the .env file.")
    st.info("📝 Create a .env file in the project root with:\n```\nGROQ_API_KEY=your_key_here\n```")
    st.stop()

if not os.path.exists("my_vector_store"):
    st.error("⚠️ Vector store not found! Please run ingestion.py first to create the vector store.")
    st.stop()

# Helper function to extract text from images using OCR
def extract_text_from_image(image_file):
    """Extract text from image using pytesseract OCR"""
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyMuPDF"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        # Extract text using PyMuPDF
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Clean up temp file
        os.unlink(tmp_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def build_recent_chat_pairs(messages, limit=5):
    """Build the most recent question-answer pairs from the current session."""
    pairs = []
    pending_question = None

    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()

        if not content:
            continue

        if role == "user":
            pending_question = content
        elif role == "assistant" and pending_question is not None:
            pairs.append({
                "question": pending_question,
                "answer": content,
                "turn_label": f"Turn {len(pairs) + 1}"
            })
            pending_question = None

    return pairs[-limit:]


def render_history_sidebar(messages):
    """Render a compact history panel in the sidebar."""
    recent_pairs = list(reversed(build_recent_chat_pairs(messages, limit=5)))

    with st.sidebar:
        st.markdown('<div class="history-badge">HISTORY</div>', unsafe_allow_html=True)
        st.markdown('<div class="history-panel-title">Recent 5 chats</div>', unsafe_allow_html=True)

        with st.expander("Open recent history", expanded=True):
            if not recent_pairs:
                st.markdown('<div class="history-empty">No previous chats yet.</div>', unsafe_allow_html=True)
            else:
                for index, item in enumerate(recent_pairs, start=1):
                    question = escape(item["question"])
                    answer = escape(item["answer"])
                    question_preview = question if len(question) <= 140 else f"{question[:140].rstrip()}..."
                    answer_preview = answer if len(answer) <= 220 else f"{answer[:220].rstrip()}..."

                    st.markdown(
                        f"""
                        <div class="history-card">
                            <div class="history-meta">
                                <span>Chat {index}</span>
                                <span>{item['turn_label']}</span>
                            </div>
                            <div class="history-question">Q: {question_preview}</div>
                            <div class="history-answer">A: {answer_preview}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource(show_spinner=False)
def load_vector_store():
    embeddings = load_embeddings()
    return FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)


def build_text_summary(text):
    """Create a detailed 5-10 line extractive summary from uploaded text."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No readable text found in this file."

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    candidates = [s.strip() for s in sentences if s.strip()]

    # Ensure summary has between 5 and 10 lines when possible.
    target_lines = max(5, min(10, len(candidates))) if candidates else 5

    if len(candidates) < 5:
        words = cleaned.split()
        chunk_size = max(12, len(words) // 5) if words else 12
        candidates = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                candidates.append(chunk)

    selected = []
    for sentence in candidates:
        selected.append(sentence)
        if len(selected) >= target_lines:
            break

    if not selected:
        return cleaned

    lines = [f"{idx + 1}. {line}" for idx, line in enumerate(selected)]
    return "\n".join(lines)


def build_llm_summary(text, file_name):
    """Generate a clean 5-10 line summary using the configured Groq model."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No readable text found in this file."

    # Limit input size to keep summarization fast and stable.
    summary_source = cleaned[:12000]

    prompt = f"""
You are a legal document summarizer.
Create a clear MAIN SUMMARY of the uploaded document in 5 to 10 numbered lines.

Rules:
1. Focus on key facts only: parties, incident/background, key legal sections, timeline, and current status.
2. Keep each line concise and readable.
3. If the text is noisy (OCR artifacts), infer carefully and ignore obvious noise.
4. Do not add facts that are not present.

File name: {file_name}
Document text:
{summary_source}
"""

    try:
        summary_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        response = summary_llm.invoke(prompt)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join([str(item) for item in content])
        content = str(content).strip()

        if content:
            return content
    except Exception:
        # Fall back to deterministic extractive summary if LLM summary fails.
        pass

    return build_text_summary(text)


LEGAL_TERMS = [
    "law", "legal", "court", "judge", "advocate", "lawyer", "attorney", "petition",
    "case", "bail", "fir", "complaint", "summons", "notice", "agreement", "contract",
    "deed", "property", "tenant", "landlord", "rent", "divorce", "custody", "maintenance",
    "alimony", "ipc", "bns", "bharatiya nyaya sanhita", "crpc", "bnss", "evidence",
    "witness", "appeal", "writ", "civil", "criminal", "plaint", "affidavit", "injunction",
    "cheque", "cheque bounce", "section", "police", "arrest", "rights", "employment",
    "termination", "labour", "salary", "consumer", "cyber", "fraud", "defamation",
    "will", "inheritance", "succession", "partnership", "company", "tax"
]


LEGAL_PHRASES = [
    "legal advice", "legal help", "law related", "court case", "file a case", "file complaint",
    "legal notice", "bail application", "property dispute", "employment dispute", "divorce case",
    "consumer complaint", "police complaint", "contract dispute", "criminal case", "civil case"
]


NON_LEGAL_RESPONSE = (
    "I can only answer legal or law-related questions. Please ask about courts, contracts, cases, "
    "rights, notices, property, family law, criminal law, or other legal matters."
)


def is_legal_related_question(question):
    """Return True when the query is clearly about law or legal matters."""
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    if not normalized:
        return False

    if any(phrase in normalized for phrase in LEGAL_PHRASES):
        return True

    word_hits = sum(1 for term in LEGAL_TERMS if re.search(rf"\b{re.escape(term)}\b", normalized))
    if word_hits >= 1:
        return True

    legal_action_patterns = [
        r"\bmy\s+(?:rights?|property|case|complaint|agreement|contract)\b",
        r"\b(?:file|draft|send|serve|respond to)\s+(?:a\s+)?(?:case|complaint|notice|petition|bail|appeal)\b",
        r"\b(?:arrest|detain|summon|evict|eviction|terminate|termination|sue|lawsuit|inherit|inheritance)\b",
    ]
    if any(re.search(pattern, normalized) for pattern in legal_action_patterns):
        return True

    return False

# Function to process uploaded documents and add to vector store
def process_and_add_documents(uploaded_files, embeddings, db):
    """Process uploaded files and add to existing vector store"""
    if not uploaded_files:
        return db, [], []
    
    documents = []
    file_summaries = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()
        
        # Process files (silent - no visible messages)
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            text = extract_text_from_image(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            continue
        
        if text.strip():
            word_count = len(text.split())
            file_summaries.append({
                "file_name": file_name,
                "file_type": file_extension,
                "word_count": word_count,
                "summary": build_llm_summary(text, file_name)
            })

            # Create document with metadata
            doc = Document(
                page_content=text,
                metadata={"source": file_name, "file_type": file_extension}
            )
            documents.append(doc)
        else:
            st.warning(f"No text extracted from {file_name}")
    
    if documents:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Add to existing vector store (silent processing)
        db.add_documents(split_docs)
        
        # Save updated vector store
        db.save_local("my_vector_store")
        st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)!")

        return db, file_summaries, split_docs

    return db, file_summaries, []

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

render_history_sidebar(st.session_state.messages)

if "uploaded_file_summaries" not in st.session_state:
    st.session_state.uploaded_file_summaries = []

if "uploaded_file_sources" not in st.session_state:
    st.session_state.uploaded_file_sources = []

if "uploaded_db" not in st.session_state:
    st.session_state.uploaded_db = None

# Initialize embeddings and vector store with HuggingFace (free & unlimited!)
try:
    embeddings = load_embeddings()
    db = load_vector_store()
    
    # Store embeddings and db in session state for reuse
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = embeddings
    if 'db' not in st.session_state:
        st.session_state.db = db
        
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
except Exception as e:
    st.error(f"❌ Error loading embeddings: {str(e)}")
    st.info("💡 Make sure you've run `python ingestion.py` to create the vector store first.")
    st.stop()

# ==================== FIRST HALF: FILE UPLOAD SECTION ====================
st.markdown('<div class="section-shell">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Document intake</div>', unsafe_allow_html=True)
st.header("📄 Upload Legal Documents")
st.markdown("Upload PDF, JPG, JPEG, or PNG files (up to 200MB per file)")

uploaded_files = st.file_uploader(
    "Choose files to upload",
    type=['pdf', 'jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="Upload legal documents to add to the knowledge base"
)

if uploaded_files:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Selected {len(uploaded_files)} file(s):**")
        for file in uploaded_files:
            file_size_mb = file.size / (1024 * 1024)
            st.write(f"- {file.name} ({file_size_mb:.2f} MB)")
    
    with col2:
        if st.button("🚀 Process Files", type="primary"):
            with st.spinner("Processing documents..."):
                updated_db, processed_summaries, split_docs = process_and_add_documents(
                    uploaded_files, 
                    st.session_state.embeddings, 
                    st.session_state.db
                )
                st.session_state.db = updated_db

                # Store file summaries for display and add source names for file-focused Q&A.
                if processed_summaries:
                    st.session_state.uploaded_file_summaries = processed_summaries
                    st.session_state.uploaded_file_sources = [
                        item["file_name"] for item in processed_summaries
                    ]

                if split_docs:
                    st.session_state.uploaded_db = FAISS.from_documents(
                        split_docs,
                        st.session_state.embeddings
                    )

                # Update retriever with new documents
                db_retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                st.rerun()

if st.session_state.uploaded_file_summaries:
    st.subheader("📌 Uploaded File Summaries")
    for item in st.session_state.uploaded_file_summaries:
        with st.expander(f"{item['file_name']} ({item['word_count']} words)", expanded=False):
            st.markdown(item["summary"].replace("\n", "  \n"))
    st.markdown("---")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-shell">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Legal Q&A</div>', unsafe_allow_html=True)
st.header("💬 Ask Legal Questions")
st.markdown("Chat with the legal assistant about your uploaded documents or existing legal knowledge")
st.markdown("---")

# Response language selector
language_options = {
    "English": "English",
    "Telugu": "Telugu",
    "Hindi": "Hindi"
}
selected_language = st.selectbox(
    "Select response language",
    options=list(language_options.keys()),
    index=0,
    help="Assistant responses will be generated in the selected language"
)
response_language = language_options[selected_language]

# Define the prompt template
prompt_template = """
<s>[INST]This is a chat template and As a legal chat bot , your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code and uploaded legal documents.
CRITICAL DOMAIN RULE: If the question is not legal or law-related, reply only with a brief refusal and do not answer the non-legal topic.
RESPONSE LANGUAGE: {response_language}
IMPORTANT: Your final answer must be entirely in the RESPONSE LANGUAGE.
MODE: {response_mode}
ANSWER FORMAT: {answer_length_instruction}
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'question', 'chat_history', 'response_language', 'response_mode', 'answer_length_instruction']
)

# Initialize the LLM with updated model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Mode selector requested by user.
selected_mode = st.selectbox(
    "Ask questions from",
    options=["GENERAL", "DOCUMENT"],
    index=0,
    help="GENERAL: legal answers from complete knowledge base. DOCUMENT: answers only from uploaded file(s)."
)

if selected_mode == "DOCUMENT" and not st.session_state.uploaded_db:
    st.warning("PLEASE UPLOAD THE DOCUMENT")

# Set up mode-specific retriever and instructions.
if selected_mode == "GENERAL":
    selected_retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    answer_length_instruction = "Provide a detailed legal answer in 10 to 15 lines."
else:
    selected_retriever = st.session_state.uploaded_db.as_retriever(search_type="similarity", search_kwargs={"k": 4}) if st.session_state.uploaded_db else st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    answer_length_instruction = "Answer only using the uploaded document context in 10 to 15 lines. If context is insufficient, state that clearly."

prompt_with_mode = prompt.partial(
    response_language=response_language,
    response_mode=selected_mode,
    answer_length_instruction=answer_length_instruction
)


def format_chat_history(messages, max_turns=4):
    """Format recent chat history for prompt context."""
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines) if lines else "No previous chat history."


def build_context_from_docs(retriever, question):
    """Retrieve relevant chunks and combine them into prompt context."""
    try:
        docs = retriever.invoke(question)
    except Exception:
        docs = retriever.get_relevant_documents(question)

    if not docs:
        return "No relevant document context found.", []

    context_parts = []
    for idx, doc in enumerate(docs[:6], start=1):
        source = doc.metadata.get("source", "unknown")
        content = str(doc.page_content or "").strip()
        if not content:
            continue
        context_parts.append(f"[Source {idx}: {source}]\n{content}")

    return "\n\n".join(context_parts) if context_parts else "No relevant document context found.", docs


def generate_answer(question, retriever, prompt_template):
    """Generate answer from retriever context + chat history using ChatGroq."""
    context, _ = build_context_from_docs(retriever, question)
    chat_history = format_chat_history(st.session_state.messages)
    formatted_prompt = prompt_template.format(
        context=context,
        chat_history=chat_history,
        question=question,
    )
    response = llm.invoke(formatted_prompt)
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "\n".join([str(item) for item in content])
    return str(content).strip() if str(content).strip() else "I could not generate a response for this question."

# Display previous messages
for message in st.session_state.messages:
    role = message.get("role")
    avatar = "⚖️" if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.write(message.get("content"))

# Input prompt
input_prompt = st.chat_input("Say something")

if input_prompt:
    # In DOCUMENT mode, allow any reasonable question about the document.
    # In GENERAL mode, enforce strict legal-only filtering.
    if selected_mode == "GENERAL" and not is_legal_related_question(input_prompt):
        with st.chat_message("assistant", avatar="⚖️"):
            st.write(NON_LEGAL_RESPONSE)
        st.session_state.messages.append({"role": "user", "content": input_prompt})
        st.session_state.messages.append({"role": "assistant", "content": NON_LEGAL_RESPONSE})
        st.stop()

    if selected_mode == "DOCUMENT" and not st.session_state.uploaded_db:
        with st.chat_message("assistant", avatar="⚖️"):
            st.write("PLEASE UPLOAD THE DOCUMENT")
        st.session_state.messages.append({"role": "assistant", "content": "PLEASE UPLOAD THE DOCUMENT"})
        st.stop()

    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Generating response..."):
            answer_text = generate_answer(input_prompt, selected_retriever, prompt_with_mode)
            message_placeholder = st.empty()
            message_placeholder.markdown(answer_text)

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)
    st.session_state.messages.append({"role": "assistant", "content": answer_text})

st.markdown('</div>', unsafe_allow_html=True)