import streamlit as st
from app.ui import build_ui

st.set_page_config(page_title="QuestRAG", layout="wide")
build_ui()