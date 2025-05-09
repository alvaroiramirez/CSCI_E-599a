import torch
from ultralytics import YOLO
import streamlit as st
import os
import tempfile


# Load model once
@st.cache_resource
def load_model_check(uploaded_model):
    """
    Load a YOLO model from a user-uploaded file and return (model, device).
    Uses a temporary file to write the uploaded buffer.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_file.write(uploaded_model.getbuffer())
        tmp_modelpath = tmp_file.name
        
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(tmp_modelpath)
    # Remove the temp file after loading
    os.remove(tmp_modelpath)
    model.to(device)
    return model, device
